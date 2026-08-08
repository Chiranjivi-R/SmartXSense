"""Unit tests for file-based forecast history store (Phase D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_engine.history.forecast_history import (
    ForecastHistoryError,
    build_history_record,
    delete_forecast_run,
    get_forecast_run,
    list_forecast_history,
    record_forecast_run,
    safe_record_from_forecast_result,
)


def test_record_and_list_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "history"
    first = build_history_record(
        domain="retail",
        model="random_forest",
        target="sales",
        forecast_type="test_evaluation",
        features_id="features_demo",
        prediction_count=10,
        metrics={"mae": 1.0, "rmse": 2.0, "mape": 0.1, "r2": 0.9},
        artifact_source="trained",
    )
    second = build_history_record(
        domain="retail",
        model="xgboost",
        target="sales",
        forecast_type="future",
        features_id="features_demo",
        prediction_count=7,
        horizon=7,
        artifact_source="loaded",
    )
    record_forecast_run(first, history_root=root)
    record_forecast_run(second, history_root=root)

    runs = list_forecast_history(history_root=root, limit=20)
    assert len(runs) == 2
    assert runs[0]["run_id"] == second["run_id"]
    assert runs[1]["run_id"] == first["run_id"]
    assert "predictions" not in runs[0]
    assert "predictions" not in json.loads((root / "runs" / f"{second['run_id']}.json").read_text(encoding="utf-8"))


def test_filters_domain_model_and_limit(tmp_path: Path) -> None:
    root = tmp_path / "history"
    for domain, model in (("retail", "random_forest"), ("electricity", "xgboost"), ("retail", "xgboost")):
        record_forecast_run(
            build_history_record(
                domain=domain,
                model=model,
                target="sales" if domain == "retail" else "demand_mw",
                forecast_type="test_evaluation",
                features_id="features_x",
                prediction_count=5,
                metrics={"rmse": 1.0},
                artifact_source="trained",
            ),
            history_root=root,
        )

    retail = list_forecast_history(domain="retail", history_root=root, limit=20)
    assert len(retail) == 2
    assert all(item["domain"] == "retail" for item in retail)

    xgb = list_forecast_history(model="xgboost", history_root=root, limit=20)
    assert len(xgb) == 2
    assert all(item["model"] == "xgboost" for item in xgb)

    limited = list_forecast_history(history_root=root, limit=1)
    assert len(limited) == 1

    with pytest.raises(ForecastHistoryError):
        list_forecast_history(history_root=root, limit=0)
    with pytest.raises(ForecastHistoryError):
        list_forecast_history(history_root=root, limit=101)


def test_get_and_delete_run(tmp_path: Path) -> None:
    root = tmp_path / "history"
    record = build_history_record(
        domain="retail",
        model="random_forest",
        target="sales",
        forecast_type="test_evaluation",
        features_id="features_demo",
        prediction_count=3,
        artifact_source="trained",
    )
    record_forecast_run(record, history_root=root)
    loaded = get_forecast_run(record["run_id"], history_root=root)
    assert loaded["run_id"] == record["run_id"]
    assert "predictions" not in loaded

    delete_forecast_run(record["run_id"], history_root=root)
    with pytest.raises(ForecastHistoryError) as exc:
        get_forecast_run(record["run_id"], history_root=root)
    assert exc.value.code == 404


def test_invalid_run_id_404(tmp_path: Path) -> None:
    with pytest.raises(ForecastHistoryError) as exc:
        get_forecast_run("../etc/passwd", history_root=tmp_path / "history")
    assert exc.value.code == 404


def test_corrupted_index_fails_gracefully(tmp_path: Path) -> None:
    root = tmp_path / "history"
    root.mkdir(parents=True)
    (root / "index.json").write_text("{not-json", encoding="utf-8")
    runs = list_forecast_history(history_root=root, limit=10)
    assert runs == []


def test_missing_history_dir_lists_empty(tmp_path: Path) -> None:
    runs = list_forecast_history(history_root=tmp_path / "missing", limit=10)
    assert runs == []


def test_rejects_auto_select_model_name() -> None:
    with pytest.raises(ForecastHistoryError):
        build_history_record(
            domain="retail",
            model="auto_select",
            target="sales",
            forecast_type="test_evaluation",
            features_id="features_demo",
            prediction_count=1,
        )


def test_index_skips_malformed_entries_and_strips_predictions(tmp_path: Path) -> None:
    root = tmp_path / "history"
    good = build_history_record(
        domain="retail",
        model="random_forest",
        target="sales",
        forecast_type="test_evaluation",
        features_id="features_demo",
        prediction_count=2,
        artifact_source="trained",
    )
    record_forecast_run(good, history_root=root)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.extend(
        [
            {"run_id": "../escape", "domain": "retail", "model": "xgboost", "forecast_type": "future"},
            {
                "run_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "domain": "retail",
                "model": "xgboost",
                "forecast_type": "future",
                "predictions": [{"x": 1}],
            },
            "broken",
        ]
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    runs = list_forecast_history(history_root=root, limit=20)
    assert any(item["run_id"] == good["run_id"] for item in runs)
    assert all("predictions" not in item for item in runs)
    assert all(item.get("run_id") != "../escape" for item in runs)


def test_safe_record_strips_predictions(tmp_path: Path) -> None:
    root = tmp_path / "history"
    result = {
        "domain": "retail",
        "model": "random_forest",
        "target": "sales",
        "prediction_count": 2,
        "test_metrics": {"mae": 1.0, "rmse": 2.0, "mape": 0.2, "r2": 0.8},
        "predictions": [{"timestamp": "2024-01-01", "actual": 1, "predicted": 1.1}] * 500,
        "artifact": {"source": "trained"},
    }
    summary = safe_record_from_forecast_result(
        result=result,
        forecast_type="test_evaluation",
        features_id="features_uploaded_abc",
        history_root=root,
    )
    assert summary is not None
    stored = get_forecast_run(summary["run_id"], history_root=root)
    assert "predictions" not in stored
    assert stored["features_id"] == "features_uploaded_abc"
    assert stored["artifact_source"] == "trained"
    assert stored["model"] == "random_forest"
