"""Safe model artifact persistence for SmartXSense forecasting models.

Artifacts are stored only under the controlled project artifacts directory.
API callers never supply absolute artifact paths — lookup is by domain, model,
features identifier, and configuration fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACT_VERSION = 1
ARTIFACT_FORMAT_PICKLE = "pickle"
ARTIFACT_FORMAT_TORCH = "torch_bundle"
ARTIFACT_FORMAT_HYBRID = "hybrid_bundle"

PERSISTENCE_SUPPORTED_MODELS = frozenset({
    "naive",
    "seasonal_naive",
    "arima",
    "random_forest",
    "xgboost",
    "trend_xgboost",
    "lstm",
    "cnn_lstm",
})

TORCH_MODELS = frozenset({"lstm", "cnn_lstm"})

DEFAULT_ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "models"

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_ARTIFACT_LOCK = threading.RLock()


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


def compute_dataset_fingerprint(features_dir: Path | str) -> str:
    """Compute a deterministic content hash fingerprint of train/validation/test parquet splits."""
    fdir = Path(features_dir).resolve()
    hasher = hashlib.sha256()
    hasher.update(fdir.name.encode("utf-8"))
    for fname in ("train.parquet", "validation.parquet", "test.parquet"):
        fpath = fdir / fname
        if fpath.exists():
            hasher.update(fname.encode("utf-8"))
            try:
                with open(fpath, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
            except OSError as err:
                raise ArtifactError(f"Unable to hash feature file {fname}: {err}") from err
    return hasher.hexdigest()[:12]


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
    dataset_fingerprint: str | None = None,
    configuration: dict[str, Any] | None,
    artifacts_root: Path | str | None = None,
) -> Path:
    """Resolve the controlled directory for a compatible artifact identity."""
    root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
    cfg = config_fingerprint(configuration)
    dom = _safe_segment(domain)
    m_key = _safe_segment(model_key)
    f_id = _safe_segment(features_id)
    domain_dir = root / dom

    if dataset_fingerprint:
        ds_fp = _safe_segment(dataset_fingerprint)
        rel = Path(dom) / f"{m_key}__{f_id}_{ds_fp}__{cfg}"
        candidate = (root / rel).resolve()
    else:
        prefix = f"{m_key}__{f_id}_"
        suffix = f"__{cfg}"
        found: Path | None = None
        if domain_dir.is_dir():
            for entry in domain_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix) and entry.name.endswith(suffix):
                    found = entry
                    break
                elif entry.is_dir() and entry.name == f"{m_key}__{f_id}__{cfg}":
                    found = entry
                    break
        if found is not None:
            candidate = found.resolve()
        else:
            rel = Path(dom) / f"{m_key}__{f_id}__{cfg}"
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
    dataset_fingerprint: str | None = None,
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
        "dataset_fingerprint": dataset_fingerprint or "",
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
    tmp_path = artifact_dir / "metadata.json.tmp"
    data = json.dumps(metadata, indent=2, sort_keys=True)
    tmp_path.write_text(data, encoding="utf-8")
    os.replace(tmp_path, _metadata_path(artifact_dir))


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
    dataset_fingerprint: str | None = None,
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

    if dataset_fingerprint is not None:
        saved_fp = metadata.get("dataset_fingerprint")
        if saved_fp and saved_fp != dataset_fingerprint:
            raise ArtifactError("Artifact dataset fingerprint mismatch (dataset content changed)")

    saved_names = list(metadata.get("feature_names") or [])
    saved_count = int(metadata.get("feature_count", -1))
    if saved_count != len(saved_names):
        raise ArtifactError("Artifact feature metadata is inconsistent")

    FORBIDDEN_KEYWORDS_IN_ARTIFACT = {
        "customer_id", "customer_name", "first_name", "last_name", "date_of_birth", "dob",
        "postal_code", "ssn", "credit_card", "customerid"
    }



    saved_names_lower = [str(n).lower().replace(" ", "_") for n in saved_names]
    for forbidden in FORBIDDEN_KEYWORDS_IN_ARTIFACT:
        if any(forbidden in n or forbidden == n for n in saved_names_lower):
            raise ArtifactError(f"Artifact contains forbidden feature '{forbidden}'; invalidating stale artifact for fresh re-training.")

    if feature_names is not None:
        req_names = list(feature_names)
        if req_names != saved_names:
            raise ArtifactError(
                f"Artifact feature schema mismatch (feature names mismatch): expected {len(req_names)} features ({req_names[:3]}...), "
                f"got {len(saved_names)} features ({saved_names[:3]}...)"
            )




    if configuration is not None:
        if _canonical_config(metadata.get("configuration")) != _canonical_config(configuration):
            raise ArtifactError("Artifact configuration mismatch")



def _serialize_torch_forecaster(model: Any, model_key: str) -> dict[str, Any]:
    if model_key == "lstm":
        if getattr(model, "_net", None) is None or getattr(model, "_preprocessor", None) is None:
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
        if getattr(model, "_net", None) is None or getattr(model, "_preprocessor", None) is None:
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


def _serialize_hybrid_forecaster(model: Any) -> dict[str, Any]:
    if not hasattr(model, "weights") or model.weights is None:
        raise ArtifactError("Hybrid model is not fitted (missing weights)")
    component_models = getattr(model, "component_models", {})
    if not component_models:
        raise ArtifactError("Hybrid model component models not available for serialization")

    serialized_components: dict[str, Any] = {}
    for comp_key, comp_model in component_models.items():
        if comp_key in TORCH_MODELS:
            serialized_components[comp_key] = _serialize_torch_forecaster(comp_model, comp_key)
        else:
            serialized_components[comp_key] = comp_model

    return {
        "model_key": "hybrid",
        "domain": getattr(model, "domain", "retail"),
        "weighting_strategy": getattr(model, "weighting_strategy", "inverse_rmse"),
        "components": list(model.components),
        "weights": dict(model.weights),
        "component_models": serialized_components,
    }


def _deserialize_hybrid_forecaster(bundle: dict[str, Any]) -> Any:
    from .hybrid import HybridForecaster

    forecaster = HybridForecaster(
        components=bundle["components"],
        domain=bundle.get("domain", "retail"),
        weighting_strategy=bundle.get("weighting_strategy", "inverse_rmse"),
    )
    forecaster.weights = bundle["weights"]
    component_models: dict[str, Any] = {}
    for comp_key, comp_val in bundle["component_models"].items():
        if comp_key in TORCH_MODELS:
            component_models[comp_key] = _deserialize_torch_forecaster(comp_val)
        else:
            component_models[comp_key] = comp_val
    forecaster.component_models = component_models
    return forecaster


def save_model_artifact(
    *,
    model: Any,
    domain: str,
    model_key: str,
    target: str,
    features_id: str,
    dataset_fingerprint: str | None = None,
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
        raise ArtifactError(f"Model '{model_key}' does not support safe artifact persistence")

    with _ARTIFACT_LOCK:
        root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
        artifact_dir = artifact_directory(
            domain=domain,
            model_key=model_key,
            features_id=features_id,
            dataset_fingerprint=dataset_fingerprint,
            configuration=configuration,
            artifacts_root=root,
        )
        _assert_within_root(artifact_dir, root)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if model_key == "hybrid":
            serialization_format = ARTIFACT_FORMAT_HYBRID
            hybrid_bundle = _serialize_hybrid_forecaster(model)
            tmp_pkl = artifact_dir / "model.pkl.tmp"
            with open(tmp_pkl, "wb") as handle:
                pickle.dump({"model_key": "hybrid", "bundle": hybrid_bundle}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_pkl, _pickle_path(artifact_dir))
            _torch_path(artifact_dir).unlink(missing_ok=True)
        elif model_key in TORCH_MODELS:
            import torch

            serialization_format = ARTIFACT_FORMAT_TORCH
            bundle = _serialize_torch_forecaster(model, model_key)
            tmp_pt = artifact_dir / "model.pt.tmp"
            torch.save(bundle, tmp_pt)
            os.replace(tmp_pt, _torch_path(artifact_dir))
            _pickle_path(artifact_dir).unlink(missing_ok=True)
        else:
            serialization_format = ARTIFACT_FORMAT_PICKLE
            tmp_pkl = artifact_dir / "model.pkl.tmp"
            with open(tmp_pkl, "wb") as handle:
                pickle.dump({"model_key": model_key, "model": model}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_pkl, _pickle_path(artifact_dir))
            _torch_path(artifact_dir).unlink(missing_ok=True)

        metadata = build_metadata(
            domain=domain,
            model=model_key,
            target=target,
            feature_names=feature_names,
            configuration=configuration,
            features_id=features_id,
            dataset_fingerprint=dataset_fingerprint,
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
    dataset_fingerprint: str | None = None,
    configuration: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    artifacts_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load a compatible artifact from the controlled artifacts directory."""
    model_key = model_key.lower().strip()
    domain = domain.lower().strip()

    if not is_persistence_supported(model_key):
        raise ArtifactError(f"Model '{model_key}' does not support artifact loading")

    with _ARTIFACT_LOCK:
        root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
        artifact_dir = artifact_directory(
            domain=domain,
            model_key=model_key,
            features_id=features_id,
            dataset_fingerprint=dataset_fingerprint,
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
            dataset_fingerprint=dataset_fingerprint,
            feature_names=feature_names,
            configuration=configuration,
        )

        fmt = metadata.get("serialization_format")
        try:
            if fmt == ARTIFACT_FORMAT_HYBRID:
                path = _pickle_path(artifact_dir)
                if not path.is_file():
                    raise ArtifactError("Hybrid pickle file missing")
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                if not isinstance(payload, dict) or "bundle" not in payload:
                    raise ArtifactError("Corrupt hybrid artifact payload")
                model = _deserialize_hybrid_forecaster(payload["bundle"])
            elif fmt == ARTIFACT_FORMAT_TORCH:
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
    dataset_fingerprint: str | None = None,
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
            dataset_fingerprint=dataset_fingerprint,
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

    if model_key == "hybrid":
        component_models = getattr(model, "component_models", {})
        val_predictions: dict[str, pd.Series] = {}
        test_predictions: dict[str, pd.Series] = {}
        for comp in model.components:
            comp_model = component_models[comp]
            comp_res = predict_with_fitted_model(
                model_key=comp,
                model=comp_model,
                domain=domain,
                target_column=target_column,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )
            val_predictions[comp] = comp_res["val_pred"]
            test_predictions[comp] = comp_res["test_pred"]

        from .hybrid import HybridForecaster

        val_pred = HybridForecaster.combine(val_predictions, model.weights)
        test_pred = HybridForecaster.combine(test_predictions, model.weights)
        return {
            "model_key": "hybrid",
            "model": model,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "feature_names": [],
            "feature_importances": None,
            "val_predictions": val_predictions,
            "test_predictions": test_predictions,
        }

    if model_key == "arima":
        val_pred = model.predict(val_df, target_column=target_column, history=train_df[target_column])
        combined = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        test_pred = model.predict(test_df, target_column=target_column, history=combined)
    elif model_key in ("naive", "seasonal_naive"):
        val_pred = model.predict(val_df, target_column=target_column, history=train_df[target_column])
        combined = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        test_pred = model.predict(test_df, target_column=target_column, history=combined)
    elif model_key in ("random_forest", "xgboost", "trend_xgboost"):
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


BENCHMARK_CACHE_VERSION = 1


def compute_benchmark_key(
    *,
    domain: str,
    features_id: str,
    dataset_fingerprint: str,
    candidate_models: list[str],
    selection_metric: str,
    configurations: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Compute a deterministic short hash key for a benchmark evaluation configuration."""
    canonical_models = sorted(str(m).lower().strip() for m in candidate_models)
    canonical_configs = {}
    if configurations:
        for m in canonical_models:
            if m in configurations:
                canonical_configs[m] = _canonical_config(configurations[m])
    payload = {
        "benchmark_version": BENCHMARK_CACHE_VERSION,
        "domain": domain.lower().strip(),
        "features_id": features_id,
        "dataset_fingerprint": dataset_fingerprint,
        "candidate_models": canonical_models,
        "selection_metric": selection_metric.lower().strip(),
        "configurations": canonical_configs,
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def benchmark_cache_path(
    *,
    domain: str,
    key: str,
    artifacts_root: Path | str | None = None,
) -> Path:
    """Resolve destination path for a benchmark cache file under controlled root."""
    root = Path(artifacts_root).resolve() if artifacts_root is not None else default_artifacts_root()
    dom = _safe_segment(domain)
    safe_key = _safe_segment(key)
    bench_dir = root / dom / "benchmarks"
    path = (bench_dir / f"benchmark_{safe_key}.json").resolve()
    _assert_within_root(path, root)
    return path


def try_load_benchmark_cache(
    *,
    domain: str,
    features_id: str,
    dataset_fingerprint: str,
    candidate_models: list[str],
    selection_metric: str,
    configurations: dict[str, dict[str, Any]] | None = None,
    artifacts_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Attempt to load a compatible cached benchmark result. Return None if missing/incompatible/corrupt."""
    try:
        key = compute_benchmark_key(
            domain=domain,
            features_id=features_id,
            dataset_fingerprint=dataset_fingerprint,
            candidate_models=candidate_models,
            selection_metric=selection_metric,
            configurations=configurations,
        )
        path = benchmark_cache_path(domain=domain, key=key, artifacts_root=artifacts_root)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if int(data.get("benchmark_version", -1)) != BENCHMARK_CACHE_VERSION:
            return None
        if data.get("domain") != domain.lower().strip():
            return None
        if data.get("dataset_fingerprint") != dataset_fingerprint:
            return None
        if data.get("selection_metric") != selection_metric.lower().strip():
            return None
        return data
    except Exception:
        # Missing, inaccessible, or corrupt cache must fall back to recomputing gracefully
        return None


def save_benchmark_cache(
    *,
    domain: str,
    features_id: str,
    dataset_fingerprint: str,
    candidate_models: list[str],
    selection_metric: str,
    ranking: list[dict[str, Any]],
    recommended_model: str,
    adaptive_selection: dict[str, Any],
    configurations: dict[str, dict[str, Any]] | None = None,
    artifacts_root: Path | str | None = None,
) -> dict[str, Any]:
    """Atomically save a benchmark result to cache under controlled root."""
    with _ARTIFACT_LOCK:
        key = compute_benchmark_key(
            domain=domain,
            features_id=features_id,
            dataset_fingerprint=dataset_fingerprint,
            candidate_models=candidate_models,
            selection_metric=selection_metric,
            configurations=configurations,
        )
        path = benchmark_cache_path(domain=domain, key=key, artifacts_root=artifacts_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark_version": BENCHMARK_CACHE_VERSION,
            "domain": domain.lower().strip(),
            "features_id": features_id,
            "dataset_fingerprint": dataset_fingerprint,
            "selection_metric": selection_metric.lower().strip(),
            "selection_split": "validation",
            "candidate_models": sorted(str(m).lower().strip() for m in candidate_models),
            "ranking": ranking,
            "recommended_model": recommended_model,
            "adaptive_selection": adaptive_selection,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = path.parent / f"{path.name}.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
        return payload
