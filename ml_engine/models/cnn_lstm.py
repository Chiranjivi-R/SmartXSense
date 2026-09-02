"""CNN-LSTM forecasting model implemented in PyTorch.

Phase 5G extends Phase 5F LSTM architecture with a 1D temporal convolution
front-end, allowing the model to extract local temporal patterns before
the LSTM processes the sequence.

Architecture:
    Input  (batch, lookback, n_features)
    Transpose -> (batch, n_features, lookback)         [for Conv1d]
    Conv1d(n_features, cnn_filters, kernel_size)
    ReLU
    Transpose -> (batch, lookback - kernel_size + 1, cnn_filters)  [for LSTM]
    LSTM(cnn_filters, lstm_units)
    Dropout
    Linear(lstm_units, 1)

Short-lookback Safety:
    kernel_size is validated against lookback at construction time.
    Pooling is intentionally omitted to avoid temporal-dim collapse on short sequences.

Reuse Policy:
    CNN-LSTM reuses Phase 5F SequencePreprocessor without modification:
    - TabularFeatureTransformer (TRAIN-only categorical/imputation/exclusion)
    - TRAIN-only feature StandardScaler
    - TRAIN-only target StandardScaler
    - Chronological sequence construction (no shuffle)
    - Split-boundary history behavior identical to LSTM

Strict Leakage & Pipeline Rules:
    1. Current target y(t) is NEVER part of its own input sequence X(t).
    2. Sequences are chronological; no shuffling.
    3. All preprocessing fitted on TRAIN only.
    4. Validation/test predictions are inverse-scaled before evaluation.
    5. Model is trained exclusively on TRAIN sequences.
    6. EarlyStopping monitors validation loss only; never test loss.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .sequence_preprocessing import SequencePreprocessor, SequencePreprocessingError


class CNNLSTMModelError(ValueError):
    """Raised when CNN-LSTM model configuration, fitting, or prediction fails."""


# Default domain-aware lookback windows (shared with LSTM)
DEFAULT_LOOKBACK_CNN_LSTM = {
    "electricity": 24,
    "retail": 7,
}


def _set_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)


class _CNNLSTMNet(nn.Module):
    """CNN-LSTM network for temporal regression.

    Tensor shapes through the network:
        x_in:     (batch, lookback, n_features)
        after permute: (batch, n_features, lookback)   -> Conv1d input (C, L)
        after conv:    (batch, cnn_filters, lookback - kernel_size + 1)
        after relu:    same
        after permute: (batch, lookback - kernel_size + 1, cnn_filters) -> LSTM input
        after lstm:    (batch, lookback - kernel_size + 1, lstm_units)
        last step:     (batch, lstm_units)
        after dropout: (batch, lstm_units)
        after linear:  (batch,)  [squeezed]
    """

    def __init__(
        self,
        n_features: int,
        cnn_filters: int,
        kernel_size: int,
        lstm_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
        )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(lstm_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        x = x.permute(0, 2, 1)           # -> (batch, n_features, lookback)
        x = self.relu(self.conv1(x))      # -> (batch, cnn_filters, L')
        x = x.permute(0, 2, 1)           # -> (batch, L', cnn_filters)
        out, _ = self.lstm(x)             # -> (batch, L', lstm_units)
        out = self.dropout(out[:, -1, :]) # last time step -> (batch, lstm_units)
        return self.linear(out).squeeze(-1)  # -> (batch,)


class CNNLSTMForecaster:
    """PyTorch CNN-LSTM forecaster with train-only preprocessing.

    Extends Phase 5F architecture with a 1D convolution front-end that
    captures local temporal patterns before the LSTM processes the sequence.

    Reuses Phase 5F SequencePreprocessor identically:
        model.fit(train_df, target_column, val_df=val_df)
        predictions = model.predict(eval_df, target_column, history_df=train_df)

    Attributes set after fit():
        n_training_sequences: Actual len(X_train) from fit (not from val_pred).
        epochs_trained: Number of epochs completed before early stopping.
        feature_names: Feature names from the fitted SequencePreprocessor.
    """

    def __init__(
        self,
        *,
        lookback: int = 24,
        cnn_filters: int = 32,
        kernel_size: int = 3,
        lstm_units: int = 64,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_seed: int = 42,
        patience: int = 5,
    ) -> None:
        # Validate kernel_size against lookback immediately
        if kernel_size < 1:
            raise CNNLSTMModelError("kernel_size must be >= 1")
        if kernel_size > lookback:
            raise CNNLSTMModelError(
                f"kernel_size ({kernel_size}) must be <= lookback ({lookback}). "
                "After convolution the temporal dimension would be <= 0."
            )

        self.lookback = lookback
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.patience = patience

        self._preprocessor: SequencePreprocessor | None = None
        self._net: _CNNLSTMNet | None = None
        self._is_fitted = False
        self.epochs_trained: int = 0
        self.n_training_sequences: int = 0
        self.best_val_loss: float | None = None
        self.target_column: str = ""

    def fit(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        val_df: pd.DataFrame | None = None,
    ) -> CNNLSTMForecaster:
        """Fit the CNN-LSTM model on training sequences.

        All preprocessing (encoding, imputation, scaling) is fitted on TRAIN only.
        Validation sequences are built using trailing TRAIN rows as boundary context.
        Early stopping monitors validation MSE; never test MSE.
        """
        _set_seeds(self.random_seed)

        if train_df is None or train_df.empty:
            raise CNNLSTMModelError("Training dataset cannot be empty")
        if target_column not in train_df.columns:
            raise CNNLSTMModelError(f"Target column '{target_column}' missing from training data")

        self.target_column = target_column

        if len(train_df) <= self.lookback:
            self.lookback = max(1, len(train_df) - 1)
            if self.kernel_size > self.lookback:
                self.kernel_size = self.lookback

        # ------------------------------------------------------------------ #
        # Step 1: Fit shared SequencePreprocessor on TRAIN only               #
        # ------------------------------------------------------------------ #
        self._preprocessor = SequencePreprocessor(target_column=target_column)
        self._preprocessor.fit(train_df)

        # ------------------------------------------------------------------ #
        # Step 2: Build training sequences                                     #
        # ------------------------------------------------------------------ #
        X_train, y_train = self._preprocessor.build_sequences(
            train_df, lookback=self.lookback
        )

        # Record actual count from the array — used by trainer for reporting
        self.n_training_sequences = len(X_train)

        # ------------------------------------------------------------------ #
        # Step 3: Build validation sequences for early stopping               #
        # Uses trailing TRAIN rows as boundary context (never val future rows) #
        # ------------------------------------------------------------------ #
        X_val: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if val_df is not None and not val_df.empty:
            try:
                history = (
                    train_df.iloc[-self.lookback:]
                    if len(train_df) >= self.lookback
                    else train_df
                )
                X_val, y_val = self._preprocessor.build_sequences(
                    val_df, lookback=self.lookback, history_df=history
                )
            except SequencePreprocessingError:
                X_val, y_val = None, None

        # ------------------------------------------------------------------ #
        # Step 4: Build network                                               #
        # ------------------------------------------------------------------ #
        n_features = self._preprocessor.n_features
        self._net = _CNNLSTMNet(
            n_features=n_features,
            cnn_filters=self.cnn_filters,
            kernel_size=self.kernel_size,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Convert training sequences to tensors (shape: (N, lookback, n_features))
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        train_ds = TensorDataset(X_t, y_t)
        # shuffle=False: preserve chronological order
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        best_state: dict | None = None
        best_val_loss = float("inf")
        patience_counter = 0

        # ------------------------------------------------------------------ #
        # Step 5: Training loop with optional early stopping                  #
        # ------------------------------------------------------------------ #
        self._net.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self._net(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()

            # Early stopping: monitor validation loss only
            if X_val is not None and y_val is not None:
                self._net.eval()
                with torch.no_grad():
                    X_vt = torch.tensor(X_val, dtype=torch.float32)
                    y_vt = torch.tensor(y_val, dtype=torch.float32)
                    val_preds = self._net(X_vt)
                    val_loss = loss_fn(val_preds, y_vt).item()
                self._net.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.epochs_trained = epoch + 1
                        break
            else:
                # No validation — save final weights, no early stopping
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}

            self.epochs_trained = epoch + 1

        # Restore best weights (or final if no val data)
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.best_val_loss = best_val_loss if best_val_loss < float("inf") else None
        self._net.eval()
        self._is_fitted = True
        return self

    def predict(
        self,
        eval_df: pd.DataFrame,
        target_column: str,
        history_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate inverse-scaled predictions aligned to eval_df index.

        Identical boundary-history policy as Phase 5F LSTMForecaster:
            - Provide trailing TRAIN rows when predicting validation
            - Provide trailing (TRAIN + VAL) rows when predicting test
            - Never provide future rows as history

        Args:
            eval_df: Evaluation partition (validation or test split).
            target_column: Target column name.
            history_df: Preceding partition for lookback context.
        """
        if not self._is_fitted or self._net is None or self._preprocessor is None:
            raise CNNLSTMModelError("CNNLSTMForecaster must be fitted before calling predict")
        if eval_df is None or eval_df.empty:
            raise CNNLSTMModelError("Evaluation dataset cannot be empty")

        # Trim history to exactly lookback rows (sufficient for boundary context)
        if history_df is not None and not history_df.empty:
            context = (
                history_df.iloc[-self.lookback:]
                if len(history_df) >= self.lookback
                else history_df
            )
        else:
            context = None

        X_eval, _ = self._preprocessor.build_sequences(
            eval_df, lookback=self.lookback, history_df=context
        )

        expected_input_size = self._net.conv1.in_channels
        if X_eval.shape[-1] != expected_input_size:
            raise CNNLSTMModelError(
                f"Input feature dimension mismatch: dataset feature matrix has {X_eval.shape[-1]} features, "
                f"but fitted CNN-LSTM model expects in_channels={expected_input_size}"
            )


        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_eval, dtype=torch.float32)
            preds_scaled = self._net(X_t).numpy()


        # Inverse-scale predictions back to original target units
        preds_original = self._preprocessor.inverse_scale_target(preds_scaled)

        # Align predictions with eval_df index
        n_sequences = len(preds_original)
        if context is not None and len(context) >= self.lookback:
            # History fully covers lookback -> all eval rows get a prediction
            target_index = eval_df.index[:n_sequences]
        else:
            # Insufficient history -> first rows skipped
            skip = self.lookback - (len(context) if context is not None else 0)
            target_index = eval_df.index[skip : skip + n_sequences]

        return pd.Series(preds_original, index=target_index, dtype=float)

    @property
    def feature_names(self) -> list[str]:
        """Feature names produced by the fitted SequencePreprocessor."""
        if self._preprocessor is None:
            return []
        return self._preprocessor.feature_names
