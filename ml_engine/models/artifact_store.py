"""Safe model artifact persistence for SmartXSense forecasting models.

Artifacts are stored only under the controlled project artifacts directory.
API callers never supply absolute artifact paths — lookup is by domain, model,
features identifier, and configuration fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACT_VERSION = 1
ARTIFACT_FORMAT_PICKLE = "pickle"
ARTIFACT_FORMAT_TORCH = "torch_bundle"

# Hybrid is intentionally excluded: fitted component models are not retained
# by HybridForecaster, so a weights-only artifact cannot safely support reuse.
PERSISTENCE_SUPPORTED_MODELS = frozenset({
    "naive",
    "seasonal_naive",
    "arima",
    "random_forest",
    "xgboost",
    "lstm",
    "cnn_lstm",
})

TORCH_MODELS = frozenset({"lstm", "cnn_lstm"})

DEFAULT_ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "models"

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ArtifactError(ValueError):
    """Raised when artifact save/load/compatibility validation fails."""


def is_persistence_supported(model_key: str) -> bool:
    """Return True when the model family supports safe train-once reuse."""
    return model_key.lower().strip() in PERSISTENCE_SUPPORTED_MODELS


def default_artifacts_root() -> Path:
    return DEFAULT_ARTIFACTS_ROOT.resolve()


def derive_features_id(features_dir: Path | str) -> str:
    """Derive a stable features identifier from the feature directory name only.

    Never trusts absolute filesystem paths as metadata — only the basename.
    """
    name = Path(features_dir).name.strip()
    if not name or name in {".", ".."}:
        raise ArtifactError("Unable to derive features identifier from features directory")
    safe = _SAFE_ID_RE.sub("_", name)
    if not safe or safe in {".", ".."}:
        raise ArtifactError("Invalid features identifier")
    return safe


def _canonical_config(configuration: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-stable configuration dict for fingerprinting/comparison."""
    if not configuration:
        return {}
    return json.loads(json.dumps(configuration, sort_keys=True, default=str))


def config_fingerprint(configuration: dict[str, Any] | None) -> str:
    """Stable short fingerprint of training configuration."""
    payload = json.dumps(_canonical_config(configuration), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _safe_segment(value: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", value.strip())
    if not cleaned or ".." in cleaned:
        raise ArtifactError(f"Unsafe artifact path segment: {value!r}")
    return cleaned


def artifact_directory(
    *,
    domain: str,
    model_key: str,
    features_id: str,
    configuration: dict[str, Any] | None,
    artifacts_root: Path | str | None = None,
) -> Path:
    """Resolve the controlled directory for a compatible artifact identity."""
    root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
    cfg = config_fingerprint(configuration)
    rel = Path(_safe_segment(domain)) / (
        f"{_safe_segment(model_key)}__{_safe_segment(features_id)}__{cfg}"
    )
    candidate = (root / rel).resolve()
    if root not in candidate.parents and candidate != root:
        raise ArtifactError("Artifact path escaped controlled artifacts root")
    return candidate


def build_metadata(
    *,
    domain: str,
    model: str,
    target: str,
    feature_names: list[str] | None,
    configuration: dict[str, Any] | None,
    features_id: str,
    validation_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
    serialization_format: str,
) -> dict[str, Any]:
    """Create artifact metadata for compatibility checks and API responses."""
    names = list(feature_names or [])
    return {
        "artifact_version": ARTIFACT_VERSION,
        "domain": domain,
        "model": model,
        "target": target,
        "feature_names": names,
        "feature_count": len(names),
        "configuration": _canonical_config(configuration),
        "features_id": features_id,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": validation_metrics or {},
        "test_metrics": test_metrics or {},
        "serialization_format": serialization_format,
        "persistence_supported": True,
    }


def _metadata_path(artifact_dir: Path) -> Path:
    return artifact_dir / "metadata.json"


def _pickle_path(artifact_dir: Path) -> Path:
    return artifact_dir / "model.pkl"


def _torch_path(artifact_dir: Path) -> Path:
    return artifact_dir / "model.pt"


def _write_metadata(artifact_dir: Path, metadata: dict[str, Any]) -> None:
    _metadata_path(artifact_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_metadata(artifact_dir: Path) -> dict[str, Any]:
    path = _metadata_path(artifact_dir)
    if not path.is_file():
        raise ArtifactError("Artifact metadata file missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ArtifactError(f"Corrupt artifact metadata: {err}") from err
    if not isinstance(data, dict):
        raise ArtifactError("Corrupt artifact metadata: expected object")
    return data


def _assert_within_root(path: Path, artifacts_root: Path) -> None:
    resolved = path.resolve()
    root = artifacts_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ArtifactError("Refusing to access artifact outside controlled directory")


def check_compatibility(
    metadata: dict[str, Any],
    *,
    domain: str,
    model: str,
    target: str,
    features_id: str,
    feature_names: list[str] | None = None,
    configuration: dict[str, Any] | None = None,
) -> None:
    """Raise ArtifactError when metadata is incompatible with the request."""
    if int(metadata.get("artifact_version", -1)) != ARTIFACT_VERSION:
        raise ArtifactError("Unsupported or mismatched artifact version")
    if metadata.get("domain") != domain:
        raise ArtifactError("Artifact domain mismatch")
    if metadata.get("model") != model:
        raise ArtifactError("Artifact model mismatch")
    if metadata.get("target") != target:
        raise ArtifactError("Artifact target mismatch")
    if metadata.get("features_id") != features_id:
        raise ArtifactError("Artifact features identifier mismatch")

    saved_names = list(metadata.get("feature_names") or [])
    saved_count = int(metadata.get("feature_count", -1))
    if saved_count != len(saved_names):
        raise ArtifactError("Artifact feature metadata is inconsistent")

    if feature_names is not None and list(feature_names) != saved_names:
        raise ArtifactError("Artifact feature names mismatch")

    if configuration is not None:
        if _canonical_config(metadata.get("configuration")) != _canonical_config(configuration):
            raise ArtifactError("Artifact configuration mismatch")


def _serialize_torch_forecaster(model: Any, model_key: str) -> dict[str, Any]:
    if model_key == "lstm":
        from .lstm import _LSTMNet

        if model._net is None or model._preprocessor is None:
            raise ArtifactError("LSTM model is not fitted")
        return {
            "model_key": "lstm",
            "state_dict": model._net.state_dict(),
            "preprocessor": model._preprocessor,
            "hyperparams": {
                "lookback": model.lookback,
                "lstm_units": model.lstm_units,
                "dropout": model.dropout,
                "epochs": model.epochs,
                "batch_size": model.batch_size,
                "learning_rate": model.learning_rate,
                "random_seed": model.random_seed,
                "patience": model.patience,
            },
            "target_column": model.target_column,
            "epochs_trained": model.epochs_trained,
            "n_training_sequences": model.n_training_sequences,
            "net_class": "_LSTMNet",
            "n_features": model._preprocessor.n_features,
        }

    if model_key == "cnn_lstm":
        if model._net is None or model._preprocessor is None:
            raise ArtifactError("CNN-LSTM model is not fitted")
        return {
            "model_key": "cnn_lstm",
            "state_dict": model._net.state_dict(),
            "preprocessor": model._preprocessor,
            "hyperparams": {
                "lookback": model.lookback,
                "cnn_filters": model.cnn_filters,
                "kernel_size": model.kernel_size,
                "lstm_units": model.lstm_units,
                "dropout": model.dropout,
                "epochs": model.epochs,
                "batch_size": model.batch_size,
                "learning_rate": model.learning_rate,
                "random_seed": model.random_seed,
                "patience": model.patience,
            },
            "target_column": model.target_column,
            "epochs_trained": model.epochs_trained,
            "n_training_sequences": model.n_training_sequences,
            "net_class": "_CNNLSTMNet",
            "n_features": model._preprocessor.n_features,
        }

    raise ArtifactError(f"Unsupported torch model key '{model_key}'")


def _deserialize_torch_forecaster(bundle: dict[str, Any]) -> Any:
    model_key = bundle.get("model_key")
    hyper = bundle["hyperparams"]
    if model_key == "lstm":
        from .lstm import LSTMForecaster, _LSTMNet

        forecaster = LSTMForecaster(**hyper)
        forecaster.target_column = bundle["target_column"]
        forecaster._preprocessor = bundle["preprocessor"]
        forecaster._net = _LSTMNet(
            n_features=int(bundle["n_features"]),
            lstm_units=forecaster.lstm_units,
            dropout=forecaster.dropout,
        )
        forecaster._net.load_state_dict(bundle["state_dict"])
        forecaster._net.eval()
        forecaster._is_fitted = True
        forecaster.epochs_trained = int(bundle.get("epochs_trained", 0))
        forecaster.n_training_sequences = int(bundle.get("n_training_sequences", 0))
        return forecaster

    if model_key == "cnn_lstm":
        from .cnn_lstm import CNNLSTMForecaster, _CNNLSTMNet

        forecaster = CNNLSTMForecaster(**hyper)
        forecaster.target_column = bundle["target_column"]
        forecaster._preprocessor = bundle["preprocessor"]
        forecaster._net = _CNNLSTMNet(
            n_features=int(bundle["n_features"]),
            cnn_filters=forecaster.cnn_filters,
            kernel_size=forecaster.kernel_size,
            lstm_units=forecaster.lstm_units,
            dropout=forecaster.dropout,
        )
        forecaster._net.load_state_dict(bundle["state_dict"])
        forecaster._net.eval()
        forecaster._is_fitted = True
        forecaster.epochs_trained = int(bundle.get("epochs_trained", 0))
        forecaster.n_training_sequences = int(bundle.get("n_training_sequences", 0))
        return forecaster

    raise ArtifactError(f"Unknown torch bundle model_key '{model_key}'")


def save_model_artifact(
    *,
    model: Any,
    domain: str,
    model_key: str,
    target: str,
    features_id: str,
    feature_names: list[str] | None,
    configuration: dict[str, Any] | None,
    validation_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
    artifacts_root: Path | str | None = None,
) -> dict[str, Any]:
    """Persist a fitted model and metadata under the controlled artifacts root."""
    model_key = model_key.lower().strip()
    domain = domain.lower().strip()

    if not is_persistence_supported(model_key):
        raise ArtifactError(
            f"Model '{model_key}' does not support safe artifact persistence "
            "(hybrid requires fitted component state which is not retained)"
        )

    root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
    artifact_dir = artifact_directory(
        domain=domain,
        model_key=model_key,
        features_id=features_id,
        configuration=configuration,
        artifacts_root=root,
    )
    _assert_within_root(artifact_dir, root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if model_key in TORCH_MODELS:
        import torch

        serialization_format = ARTIFACT_FORMAT_TORCH
        bundle = _serialize_torch_forecaster(model, model_key)
        torch.save(bundle, _torch_path(artifact_dir))
        # Remove stale pickle if present from a previous format
        _pickle_path(artifact_dir).unlink(missing_ok=True)
    else:
        serialization_format = ARTIFACT_FORMAT_PICKLE
        with open(_pickle_path(artifact_dir), "wb") as handle:
            pickle.dump({"model_key": model_key, "model": model}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _torch_path(artifact_dir).unlink(missing_ok=True)

    metadata = build_metadata(
        domain=domain,
        model=model_key,
        target=target,
        feature_names=feature_names,
        configuration=configuration,
        features_id=features_id,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        serialization_format=serialization_format,
    )
    _write_metadata(artifact_dir, metadata)

    return {
        "artifact_dir": str(artifact_dir.as_posix()),
        "metadata": metadata,
        "relative_key": f"{domain}/{artifact_dir.name}",
    }


def load_model_artifact(
    *,
    domain: str,
    model_key: str,
    target: str,
    features_id: str,
    configuration: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    artifacts_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load a compatible artifact from the controlled artifacts directory."""
    model_key = model_key.lower().strip()
    domain = domain.lower().strip()

    if not is_persistence_supported(model_key):
        raise ArtifactError(f"Model '{model_key}' does not support artifact loading")

    root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
    artifact_dir = artifact_directory(
        domain=domain,
        model_key=model_key,
        features_id=features_id,
        configuration=configuration,
        artifacts_root=root,
    )
    _assert_within_root(artifact_dir, root)

    if not artifact_dir.is_dir():
        raise ArtifactError("Artifact directory missing")

    metadata = _read_metadata(artifact_dir)
    check_compatibility(
        metadata,
        domain=domain,
        model=model_key,
        target=target,
        features_id=features_id,
        feature_names=feature_names,
        configuration=configuration,
    )

    fmt = metadata.get("serialization_format")
    try:
        if fmt == ARTIFACT_FORMAT_TORCH:
            import torch

            path = _torch_path(artifact_dir)
            if not path.is_file():
                raise ArtifactError("Torch model weights missing")
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            model = _deserialize_torch_forecaster(bundle)
        elif fmt == ARTIFACT_FORMAT_PICKLE:
            path = _pickle_path(artifact_dir)
            if not path.is_file():
                raise ArtifactError("Pickle model file missing")
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or "model" not in payload:
                raise ArtifactError("Corrupt pickle artifact payload")
            if payload.get("model_key", model_key) != model_key:
                raise ArtifactError("Pickle artifact model key mismatch")
            model = payload["model"]
        else:
            raise ArtifactError(f"Unknown serialization format '{fmt}'")
    except ArtifactError:
        raise
    except Exception as err:
        raise ArtifactError(f"Failed to deserialize artifact: {err}") from err

    return {
        "model": model,
        "metadata": metadata,
        "artifact_dir": str(artifact_dir.as_posix()),
        "relative_key": f"{domain}/{artifact_dir.name}",
    }


def try_load_compatible_artifact(
    *,
    domain: str,
    model_key: str,
    target: str,
    features_id: str,
    configuration: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    artifacts_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return loaded artifact dict or None when missing/incompatible/corrupt."""
    try:
        return load_model_artifact(
            domain=domain,
            model_key=model_key,
            target=target,
            features_id=features_id,
            configuration=configuration,
            feature_names=feature_names,
            artifacts_root=artifacts_root,
        )
    except ArtifactError:
        return None


def predict_with_fitted_model(
    *,
    model_key: str,
    model: Any,
    domain: str,
    target_column: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate validation/test predictions from an already-fitted model."""
    model_key = model_key.lower().strip()

    if model_key == "arima":
        val_pred = model.predict(val_df, target_column=target_column, history=train_df[target_column])
        combined = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        test_pred = model.predict(test_df, target_column=target_column, history=combined)
    elif model_key in ("naive", "seasonal_naive"):
        val_pred = model.predict(val_df, target_column=target_column, history=train_df[target_column])
        combined = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        test_pred = model.predict(test_df, target_column=target_column, history=combined)
    elif model_key in ("random_forest", "xgboost"):
        val_pred = model.predict(val_df)
        test_pred = model.predict(test_df)
    elif model_key in ("lstm", "cnn_lstm"):
        val_pred = model.predict(val_df, target_column=target_column, history_df=train_df)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        test_pred = model.predict(test_df, target_column=target_column, history_df=combined_df)
    else:
        raise ArtifactError(f"Cannot predict with unsupported persisted model '{model_key}'")

    feature_names: list[str] = []
    feature_importances = None
    if hasattr(model, "feature_names"):
        feature_names = list(model.feature_names or [])
    if hasattr(model, "get_feature_importances"):
        try:
            feature_importances = model.get_feature_importances()
        except Exception:
            feature_importances = None

    return {
        "model_key": model_key,
        "model": model,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "training_sequences": getattr(model, "n_training_sequences", None),
        "validation_sequences": len(val_pred) if model_key in TORCH_MODELS else None,
        "test_sequences": len(test_pred) if model_key in TORCH_MODELS else None,
        "epochs_trained": getattr(model, "epochs_trained", None),
    }
