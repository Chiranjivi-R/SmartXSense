"""LSTM forecasting model implemented in PyTorch.

TensorFlow/Keras is NOT compatible with Python 3.14.
PyTorch 2.13.0 ships cp314 wheels and is the supported alternative.

Architecture: Input -> LSTM -> Dropout -> Linear(1)

Strict Leakage & Pipeline Rules:
1. Current target y(t) is NEVER part of its own input sequence X(t).
2. Sequences are chronological; no shuffling.
3. All preprocessing (encoding, scaling, imputation) fitted on TRAIN only.
4. Validation/test predictions are inverse-scaled before metric computation.
5. Model is trained exclusively on TRAIN sequences.
6. EarlyStopping monitors validation loss only; never test loss.
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .sequence_preprocessing import SequencePreprocessor, SequencePreprocessingError


class LSTMModelError(ValueError):
    """Raised when LSTM model fitting or prediction fails."""


# Default lookback windows per domain
DEFAULT_LOOKBACK = {
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


class _LSTMNet(nn.Module):
    """Minimal LSTM network: LSTM -> Dropout -> Linear(1)."""

    def __init__(self, n_features: int, lstm_units: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(lstm_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last time step
        return self.linear(out).squeeze(-1)


class LSTMForecaster:
    """PyTorch LSTM forecaster with train-only preprocessing.

    Usage matches the Phase 5D/5E forecaster interface:
        model.fit(train_df, target_column, val_df=val_df)
        predictions = model.predict(eval_df, target_column, history_df=train_df)

    Target history policy:
        Phase 5C already provides leakage-safe lag features (e.g. lag_1, lag_24).
        These are included in the feature matrix as past-target proxies.
        The raw target column is excluded from the contemporaneous feature matrix.
        y(t) is NEVER in X(t).
    """

    def __init__(
        self,
        *,
        lookback: int = 24,
        lstm_units: int = 64,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_seed: int = 42,
        patience: int = 5,
    ) -> None:
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.patience = patience

        self._preprocessor: SequencePreprocessor | None = None
        self._net: _LSTMNet | None = None
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
    ) -> LSTMForecaster:
        """Fit the LSTM model on training sequences."""
        _set_seeds(self.random_seed)

        if train_df is None or train_df.empty:
            raise LSTMModelError("Training dataset cannot be empty")
        if target_column not in train_df.columns:
            raise LSTMModelError(f"Target column '{target_column}' missing from training data")

        self.target_column = target_column

        # Fit sequence preprocessor on TRAIN only
        self._preprocessor = SequencePreprocessor(target_column=target_column)
        self._preprocessor.fit(train_df)

        # Build training sequences
        X_train, y_train = self._preprocessor.build_sequences(
            train_df, lookback=self.lookback
        )
        # Record actual count from the training array; used by trainer for reporting
        self.n_training_sequences = len(X_train)

        # Build validation sequences using trailing train history for boundary context
        X_val: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if val_df is not None and not val_df.empty:
            try:
                # Provide last `lookback` rows of train as context for first val sequences
                history = train_df.iloc[-self.lookback:] if len(train_df) >= self.lookback else train_df
                X_val, y_val = self._preprocessor.build_sequences(
                    val_df, lookback=self.lookback, history_df=history
                )
            except SequencePreprocessingError:
                X_val, y_val = None, None

        # Build network
        n_features = self._preprocessor.n_features
        self._net = _LSTMNet(
            n_features=n_features,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Convert to tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        best_state: dict | None = None
        best_val_loss = float("inf")
        patience_counter = 0

        self._net.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self._net(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()

            # EarlyStopping on validation loss if available
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
                # No validation data — no early stopping; track all epochs
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}

            self.epochs_trained = epoch + 1

        # Restore best weights
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

        Args:
            eval_df: Evaluation partition (validation or test split).
            target_column: Target column name.
            history_df: Preceding partition (e.g. train when predicting validation,
                        or train+validation when predicting test). Used to provide
                        lookback context for the first sequences.
        """
        if not self._is_fitted or self._net is None or self._preprocessor is None:
            raise LSTMModelError("LSTMForecaster must be fitted before calling predict")
        if eval_df is None or eval_df.empty:
            raise LSTMModelError("Evaluation dataset cannot be empty")

        # Provide boundary history context
        if history_df is not None and not history_df.empty:
            context = history_df.iloc[-self.lookback:] if len(history_df) >= self.lookback else history_df
        else:
            context = None

        X_eval, _ = self._preprocessor.build_sequences(
            eval_df, lookback=self.lookback, history_df=context
        )

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_eval, dtype=torch.float32)
            preds_scaled = self._net(X_t).numpy()

        # Inverse-scale predictions back to original units
        preds_original = self._preprocessor.inverse_scale_target(preds_scaled)

        # Align with eval_df index: sequences skip the first `lookback` rows
        # when no history is provided, or cover all rows when history covers lookback.
        n_sequences = len(preds_original)
        if context is not None and len(context) >= self.lookback:
            # History fully covers lookback -> all eval rows get a prediction
            target_index = eval_df.index[:n_sequences]
        else:
            # Not enough history -> first rows skipped
            skip = self.lookback - (len(context) if context is not None else 0)
            target_index = eval_df.index[skip : skip + n_sequences]

        return pd.Series(preds_original, index=target_index, dtype=float)

    @property
    def feature_names(self) -> list[str]:
        if self._preprocessor is None:
            return []
        return self._preprocessor.feature_names
