"""Deterministic, evidence-bound forecast report summaries.

This deliberately consumes structured facts instead of an LLM.  A future
natural-language provider can be placed behind ``build_forecast_summary``
without changing forecasting or history persistence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _direction(first: float, last: float) -> str:
    return "increased" if last > first else "decreased" if last < first else "remained stable"


def _timestamp_label(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp.strftime("%Y-%m-%d %H:%M UTC") if not pd.isna(timestamp) else None


def _group_extremes(frame: pd.DataFrame, group: str, value: str, prefix: str) -> dict[str, Any]:
    if group not in frame or value not in frame or frame[group].nunique(dropna=True) < 2:
        return {}
    grouped = frame.groupby(group, dropna=True)[value].mean().dropna()
    if len(grouped) < 2:
        return {}
    return {f"{prefix}_peak": str(grouped.idxmax()), f"{prefix}_low": str(grouped.idxmin())}


def derive_forecast_facts(result: dict[str, Any], *, context_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Extract auditable aggregates from forecast rows and optional test facts.

    Source rows are deliberately not returned or persisted.  Every fact is
    omitted unless its supporting values and timestamp/category field exist.
    """
    rows = result.get("predictions") if isinstance(result.get("predictions"), list) else []
    values = [float(row["predicted"]) for row in rows if isinstance(row, dict) and isinstance(row.get("predicted"), (int, float))]
    if len(values) < 2:
        return {}
    facts: dict[str, Any] = {
        "forecast_direction": _direction(values[0], values[-1]),
        "forecast_start": values[0], "forecast_end": values[-1],
        "forecast_peak": max(values), "forecast_low": min(values),
    }
    prediction_frame = pd.DataFrame(rows)
    if "timestamp" in prediction_frame:
        timestamps = pd.to_datetime(prediction_frame["timestamp"], utc=True, errors="coerce")
        if timestamps.notna().all():
            peak_index, low_index = values.index(max(values)), values.index(min(values))
            facts["forecast_peak_period"] = _timestamp_label(timestamps.iloc[peak_index])
            facts["forecast_low_period"] = _timestamp_label(timestamps.iloc[low_index])
    if result.get("forecast_type") == "future":
        return facts
    actuals = [float(row["actual"]) for row in rows if isinstance(row, dict) and isinstance(row.get("actual"), (int, float))]
    if len(actuals) >= 4:
        midpoint = len(actuals) // 2
        earlier, recent = sum(actuals[:midpoint]) / midpoint, sum(actuals[midpoint:]) / (len(actuals) - midpoint)
        facts["recent_trend"] = _direction(earlier, recent)
        facts["recent_vs_previous_percent"] = ((recent - earlier) / abs(earlier) * 100) if earlier else None
        facts["observed_peak"] = max(actuals)
        facts["observed_low"] = min(actuals)
        actual_frame = prediction_frame.loc[prediction_frame.get("actual").apply(lambda value: isinstance(value, (int, float)))].copy()
        if "timestamp" in actual_frame and not actual_frame.empty:
            actual_frame["_timestamp"] = pd.to_datetime(actual_frame["timestamp"], utc=True, errors="coerce")
            actual_frame["_actual"] = pd.to_numeric(actual_frame["actual"], errors="coerce")
            actual_frame = actual_frame.dropna(subset=["_timestamp", "_actual"])
            if not actual_frame.empty:
                facts["observed_peak_period"] = _timestamp_label(actual_frame.loc[actual_frame["_actual"].idxmax(), "_timestamp"])
                facts["observed_low_period"] = _timestamp_label(actual_frame.loc[actual_frame["_actual"].idxmin(), "_timestamp"])
                if result.get("domain") == "electricity":
                    actual_frame["_hour"] = actual_frame["_timestamp"].dt.hour
                    facts.update(_group_extremes(actual_frame, "_hour", "_actual", "hourly"))
                    actual_frame["_day_type"] = actual_frame["_timestamp"].dt.dayofweek.ge(5).map({True: "weekend", False: "weekday"})
                    facts.update(_group_extremes(actual_frame, "_day_type", "_actual", "day_type"))

    if context_frame is not None and not context_frame.empty:
        context = context_frame.copy()
        target = str(result.get("target") or "")
        if target in context:
            context["_target"] = pd.to_numeric(context[target], errors="coerce")
            context = context.dropna(subset=["_target"])
            category = next((column for column in context.columns if str(column).lower() == "category"), None)
            if result.get("domain") == "retail" and category:
                facts.update(_group_extremes(context, category, "_target", "category"))
            if result.get("domain") == "retail" and "timestamp" in context:
                context["_month"] = pd.to_datetime(context["timestamp"], utc=True, errors="coerce").dt.month
                facts.update(_group_extremes(context.dropna(subset=["_month"]), "_month", "_target", "seasonal_month"))
    return facts


def _number(value: Any, digits: int = 2) -> str | None:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return None


def build_forecast_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Produce a concise summary from actual stored run metadata only."""
    domain = str(record.get("domain") or "forecasting").capitalize()
    model = str(record.get("model") or "model").replace("_", " ").title()
    forecast_type = str(record.get("forecast_type") or "test_evaluation")
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    facts: dict[str, Any] = {
        "domain": domain,
        "model": model,
        "forecast_type": forecast_type,
        "prediction_count": record.get("prediction_count"),
        "selection_method": record.get("selection_method"),
        "artifact_source": record.get("artifact_source"),
    }
    derived = record.get("summary_facts") if isinstance(record.get("summary_facts"), dict) else {}
    facts.update(derived)

    insights: list[str] = []
    if forecast_type == "future":
        horizon = record.get("horizon")
        points = record.get("prediction_count")
        summary = (
            f"SmartXSense generated a model-based future {domain.lower()} scenario using {model}"
            f" for {horizon} forecast steps." if horizon is not None else
            f"SmartXSense generated a model-based future {domain.lower()} scenario using {model}."
        )
        interpretation = (
            f"The scenario contains {points} forecast points; actual values are not available for future forecasts."
            if points is not None else "Actual values are not available for future forecasts."
        )
        if derived.get("forecast_direction"):
            interpretation += f" Across the requested horizon, the model scenario {derived['forecast_direction']} from its first to last forecast point."
            insights.append(f"Future forecast trend: {derived['forecast_direction']}.")
        if derived.get("forecast_peak_period"):
            insights.append(f"Forecast peak period: {derived['forecast_peak_period']}.")
        if derived.get("forecast_low_period"):
            insights.append(f"Forecast low-demand period: {derived['forecast_low_period']}.")
        return {"facts": facts, "summary": summary, "interpretation": interpretation, "insights": insights}

    metric_parts = []
    for label, key, suffix in (("RMSE", "rmse", ""), ("MAPE", "mape", "%"), ("R²", "r2", "")):
        value = _number(metrics.get(key))
        if value is not None:
            if key == "mape":
                raw = float(metrics[key])
                value = _number(raw * 100 if raw <= 1 else raw) or value
            metric_parts.append(f"{label} of {value}{suffix}")
            facts[key] = metrics[key]
    detail = ", ".join(metric_parts) if metric_parts else "recorded held-out evaluation metrics"
    summary = f"SmartXSense evaluated {domain.lower()} demand using {model} on the held-out test dataset, achieving {detail}."
    mape = metrics.get("mape")
    if _number(mape) is not None:
        percentage = float(mape) * 100 if float(mape) <= 1 else float(mape)
        interpretation = f"Average percentage error on the held-out test data was approximately {percentage:.2f}%. Forecast points and validation benchmark are retained for audit and export."
    elif record.get("forecast_data", {}).get("points"):
        interpretation = "Forecast observations, validation benchmark, and model performance metrics are retained for audit and export."
    else:
        interpretation = "Forecast run metadata is retained for audit and reporting."
    if derived.get("recent_trend"):
        comparison = derived.get("recent_vs_previous_percent")
        detail = f" Recent observed demand {derived['recent_trend']} versus the preceding sampled period."
        if isinstance(comparison, (int, float)):
            detail = f" Recent observed demand {derived['recent_trend']} by approximately {abs(comparison):.1f}% versus the preceding sampled period."
        interpretation += detail
        insights.append(f"Recent trend: {derived['recent_trend']}.")
    if derived.get("recent_vs_previous_percent") is not None:
        insights.append(f"Previous-period comparison: {abs(float(derived['recent_vs_previous_percent'])):.1f}% {derived['recent_trend']}.")
    if derived.get("observed_peak_period"):
        insights.append(f"Peak period: {derived['observed_peak_period']}.")
    if derived.get("observed_low_period"):
        insights.append(f"Low-demand period: {derived['observed_low_period']}.")
    if derived.get("hourly_peak") is not None:
        insights.append(f"Hourly pattern: highest available average at {derived['hourly_peak']}:00; lowest at {derived['hourly_low']}:00.")
    if derived.get("day_type_peak") is not None:
        insights.append(f"Weekday/weekend pattern: higher available average on {derived['day_type_peak']}s.")
    if derived.get("category_peak") is not None:
        insights.append(f"Highest category: {derived['category_peak']}; lowest category: {derived['category_low']}.")
    if derived.get("seasonal_month_peak") is not None:
        insights.append(f"Available seasonal pattern: highest average in month {derived['seasonal_month_peak']}; lowest in month {derived['seasonal_month_low']}.")
    return {"facts": facts, "summary": summary, "interpretation": interpretation, "insights": insights}
