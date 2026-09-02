"""LLM Explanation Service for SmartXSense.

Routing Hierarchy:
1. Groq (Primary) - openai/gpt-oss-120b
2. Ollama (Secondary Fallback) - qwen3:8b
3. Local Deterministic Generator (Tertiary Fallback)

Guarantees:
- Server-side API keys only; never exposed to client.
- Forecasting pipeline never fails if LLM is unavailable.
- Explains calculated metrics strictly without metric invention.
- Strictly adheres to R² variance interpretation, MAPE scale context,
  transactional duplicate timestamp aggregation, feature governance (no PII),
  non-causal SHAP language, and dataset identity context.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import requests

logger = logging.getLogger(__name__)

# Excluded keywords for PII and identifier feature governance
FORBIDDEN_DRIVER_KEYWORDS = {
    "customer_id", "customer_name", "first_name", "last_name", "email", "phone",
    "mobile", "address", "postal_code", "zipcode", "zip_code", "pincode",
    "order_id", "invoice_no", "invoice_id", "product_id", "product_name",
    "sku_id", "user_id", "index", "unnamed_0", "date_of_birth", "dob"
}


def _get_config_var(key: str, default: str | None = None) -> str | None:
    """Retrieve config variable from Flask current_app or os.getenv."""
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            val = current_app.config.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    val = os.getenv(key, default)
    return str(val).strip() if val is not None and str(val).strip() else default


def _build_system_prompt() -> str:
    return """You are SmartXSense AI, a professor and domain expert explaining time-series forecasting results in a viva exam setting.

Your explanation MUST follow this exact 5-section markdown structure:

### 🧠 AI Forecast Summary
A 2–3 sentence overall interpretation of the forecast trend and model quality.

### ⚠️ Situation
Describe what the calculated forecast indicates (e.g. demand increasing, demand decreasing, demand stable, demand highly variable, or peak periods) based strictly on actual prediction metrics.

### 🔎 Why / Possible Causes
Explain the key influencing factors driving the predictions using the provided SHAP features (e.g., recent weekly lag patterns, short-term rolling averages). Do NOT invent features or external causes not supported by the dataset.

### 💡 Recommended Actions / Remedies
Provide actionable domain guidance:
- For RETAIL: If demand is increasing, recommend inventory replenishment; if decreasing, advise avoiding excess stock accumulation.
- For ELECTRICITY: If peak demand is projected, recommend generation capacity preparation and reserve monitoring.
- For POOR/NEGATIVE R²: Recommend model review and retraining.

### 🛡️ Precautions
Advise monitoring actual incoming values against predictions and triggering model review if large deviations occur.

STRICT COMPLIANCE RULES:
1. NEVER use mechanical software jargon ("user selected model", "the API", "the service", "the application", "fallback", "lacks quantitative...").
2. R² INTERPRETATION:
   - Never call R² "accuracy".
   - If R² is positive (e.g., 0.84), state: "The model explains approximately 84% of the variation in the unseen test data."
   - If R² is negative, state: "The negative R² indicates that the model did not explain the variation in the unseen test data well enough. Therefore, the current forecast quality needs improvement."
3. DATA QUALITY COMPLIANCE:
   - ONLY mention duplicate timestamp aggregation if duplicate_timestamps_handled is TRUE.
   - ONLY mention duplicate row removal if duplicate_rows_removed > 0.
   - ONLY mention missing value handling if missing_values_handled > 0.
4. Keep the explanation under 300 words."""


def _build_user_prompt(context: dict[str, Any]) -> str:
    dataset_name = context.get("dataset_name") or context.get("dataset_id") or "Dataset"
    dataset_id = context.get("dataset_id", "unknown_dataset")
    domain = context.get("domain", "retail").lower()
    target = context.get("target", "target")
    selected_model = context.get("selected_model", "Random Forest")
    selection_reason = context.get("model_selection_reason") or "best validation performance"
    forecast_trend = context.get("forecast_trend", "stable")

    metrics = context.get("metrics", {})
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    mape_raw = metrics.get("mape")
    r2 = metrics.get("r2")
    mape_formatted = None
    if isinstance(mape_raw, (int, float)):
        mape_val = mape_raw * 100 if mape_raw <= 1.0 else mape_raw
        mape_formatted = f"{mape_val:.2f}%"

    baseline = context.get("baseline_metrics", {})
    baseline_rmse = baseline.get("naive_rmse")
    improvement_pct = baseline.get("model_improvement_pct")

    data_quality = context.get("data_quality", {})
    shap_features = context.get("shap_features", [])

    return f"""Context for Dataset '{dataset_name}' (ID: {dataset_id}):

Domain: {domain.upper()} | Target Variable: {target}
Selected Model: {selected_model} (Reason: {selection_reason})
Calculated Forecast Trend: {forecast_trend}

Held-Out Test Set Metrics:
- RMSE: {rmse}
- MAE: {mae}
- MAPE: {mape_formatted or mape_raw}
- R²: {r2}

Baseline Benchmark:
- Naive Baseline RMSE: {baseline_rmse}
- Model Improvement: {improvement_pct}%

Actual Data Quality Metadata:
- Duplicate Rows Removed: {data_quality.get('duplicate_rows_removed', 0)}
- Duplicate Timestamps Aggregated: {data_quality.get('duplicate_timestamps_handled', False)}
- Missing Values Handled: {data_quality.get('missing_values_handled', 0)}

Top Verified SHAP Drivers: {shap_features[:5]}

Generate the structured 5-section viva explanation following all compliance guidelines."""


def check_ollama_model_available(base_url: str, model_name: str) -> bool:
    """Verify if the configured Ollama model is installed and accessible locally."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            installed_names = [m.get("name", "").lower() for m in models]
            target_clean = model_name.lower().strip()
            
            # Check exact match or tag match e.g. "qwen3:8b" in "qwen3:8b:latest"
            if any(target_clean == name or name.startswith(f"{target_clean}:") for name in installed_names):
                return True
        logger.warning(f"Ollama fallback model {model_name} is not available.")
        return False
    except Exception as err:
        logger.warning(f"Ollama fallback model {model_name} is not available. Connection error: {err}")
        return False


def call_groq_api(api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str | None:
    """Call Groq Cloud Chat Completions API securely using configured GROQ_MODEL."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                text = choices[0]["message"]["content"].strip()
                if text:
                    return text
        else:
            logger.warning(f"Groq API returned HTTP {resp.status_code} for model '{model_name}': {resp.text[:200]}")
    except Exception as err:
        logger.warning(f"Groq API request failed for model '{model_name}': {err}")
    return None


def call_ollama_api(base_url: str, model_name: str, system_prompt: str, user_prompt: str) -> str | None:
    """Call local Ollama Chat API."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 450
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get("message", {})
            content = msg.get("content", "").strip() or msg.get("thinking", "").strip()
            if content:
                return content
        else:
            logger.warning(f"Ollama API returned HTTP {resp.status_code}")
    except Exception as err:
        logger.warning(f"Ollama API request failed: {err}")
    return None


def generate_deterministic_explanation(context: dict[str, Any]) -> str:
    """Generate high-integrity, professor-friendly deterministic 5-section viva explanation."""
    domain = (context.get("domain") or "retail").lower()
    target = context.get("target") or "target"
    model = context.get("selected_model") or "Random Forest"
    reason = context.get("model_selection_reason") or "best validation performance"
    trend = context.get("forecast_trend") or "stable"

    metrics = context.get("metrics", {})
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    mape = metrics.get("mape")
    r2 = metrics.get("r2")

    baseline = context.get("baseline_metrics", {})
    imp_pct = baseline.get("model_improvement_pct")

    # Section 1: Summary
    if r2 is not None and isinstance(r2, (int, float)) and r2 < 0:
        summary = f"Demand for {target} shows notable variability across the forecast horizon. While {model} was selected based on validation performance, test evaluation indicates that current forecast quality needs improvement."
    else:
        summary = f"Demand for {target} is projected to follow a {trend.replace('_', ' ')} trajectory over the forecast period. The {model} model captures primary historical demand patterns while maintaining stable error bounds."

    # Section 2: Situation
    situation_lines = []
    if trend == "increasing":
        situation_lines.append(f"The forecast indicates an upward trend in {target} demand over upcoming intervals.")
    elif trend == "decreasing":
        situation_lines.append(f"The forecast indicates a downward trend in {target} demand over upcoming intervals.")
    elif trend == "highly_variable":
        situation_lines.append(f"The forecast indicates high fluctuation in {target} demand with periodic peaks.")
    else:
        situation_lines.append(f"The forecast indicates stable {target} demand with moderate periodic fluctuations.")

    if rmse is not None and isinstance(rmse, (int, float)) and mae is not None and isinstance(mae, (int, float)):
        situation_lines.append(f"On the held-out test evaluation data, the model achieved an RMSE of {rmse:,.2f} and an MAE of {mae:,.2f}.")

    if r2 is not None and isinstance(r2, (int, float)):
        if r2 > 0:
            situation_lines.append(f"The model explains approximately {r2 * 100:.1f}% of the variation in the unseen test data.")
            if imp_pct is not None and isinstance(imp_pct, (int, float)) and imp_pct > 0:
                situation_lines.append(f"This reflects a {imp_pct:.1f}% improvement over the naive baseline prediction.")
        else:
            situation_lines.append(f"The negative R² ({r2:.2f}) indicates that the model did not explain the variation in the unseen test data well enough. Therefore, the current forecast quality needs improvement.")

    # Section 3: Why / Possible Causes
    shap_features = context.get("shap_features", [])
    drivers = []
    for sf in shap_features:
        f_name = sf.get("feature") if isinstance(sf, dict) else str(sf)
        if f_name and not any(k in str(f_name).lower() for k in FORBIDDEN_DRIVER_KEYWORDS):
            drivers.append(str(f_name))

    why_lines = []
    if drivers:
        why_lines.append(f"The forecast is primarily influenced by recent demand signals, including {', '.join(drivers[:3])}.")
        why_lines.append("These features capture short-term momentum and historical seasonal lag dependencies.")
    else:
        why_lines.append("The forecast relies on historical target lag values and rolling moving averages, which provide steady predictive baseline signals.")

    # Section 4: Recommended Actions
    remedies_lines = []
    if r2 is not None and isinstance(r2, (int, float)) and r2 < 0:
        remedies_lines.append("Given the negative R² evaluation, we recommend reviewing feature engineering specifications and retraining with additional historical depth before committing major operational resources.")
    elif domain == "electricity":
        if trend in ("increasing", "highly_variable"):
            remedies_lines.append("During predicted peak-demand periods, prepare additional generation capacity, maintain sufficient reserve capacity, and monitor grid load closely.")
        else:
            remedies_lines.append("Maintain standard generation schedules and monitor reserve capacity during expected load transitions.")
    else: # Retail
        if trend == "increasing":
            remedies_lines.append("Increase inventory for high-demand products and plan stock replenishment in advance to prevent stockouts.")
        elif trend == "decreasing":
            remedies_lines.append("Adjust procurement down to avoid excessive inventory accumulation and holding cost overheads.")
        else:
            remedies_lines.append("Maintain baseline safety stock levels and monitor weekly order replenishment signals.")

    # Section 5: Precautions
    precaution_lines = [
        "Forecasts should be continuously monitored against newly observed actual values.",
        "Large deviations or unexpected external market changes should trigger immediate model review and retraining."
    ]

    # Conditional Data Quality note
    dq = context.get("data_quality", {})
    dq_notes = []
    if dq.get("duplicate_timestamps_handled"):
        dq_notes.append("Multiple transaction records with identical timestamps were automatically aggregated into regular forecasting frequencies.")
    if dq.get("duplicate_rows_removed", 0) > 0:
        dq_notes.append(f"Cleaned {dq['duplicate_rows_removed']} duplicate rows during preparation.")
    if dq_notes:
        precaution_lines.append(" ".join(dq_notes))

    return f"""### 🧠 AI Forecast Summary
{summary}

### ⚠️ Situation
{" ".join(situation_lines)}

### 🔎 Why / Possible Causes
{" ".join(why_lines)}

### 💡 Recommended Actions / Remedies
{" ".join(remedies_lines)}

### 🛡️ Precautions
{" ".join(precaution_lines)}"""


def build_structured_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Construct structured UI card components from context and metrics."""
    metrics = context.get("metrics", {})
    baseline = context.get("baseline_metrics", {})
    dq = context.get("data_quality", {})
    shap_features = context.get("shap_features", [])

    drivers = []
    for sf in shap_features:
        f_name = sf.get("feature") if isinstance(sf, dict) else str(sf)
        if f_name and not any(k in str(f_name).lower() for k in FORBIDDEN_DRIVER_KEYWORDS):
            drivers.append(str(f_name))

    r2_val = metrics.get("r2")
    r2_desc = "N/A"
    if isinstance(r2_val, (int, float)):
        if r2_val > 0:
            r2_desc = f"Explains {r2_val * 100:.1f}% of variance"
        elif abs(r2_val) < 0.05:
            r2_desc = "Baseline level (R² ≈ 0)"
        else:
            r2_desc = f"Negative generalization (R² = {r2_val:.2f})"

    # Build conditional data quality list — NO FALSE STATEMENTS
    dq_warnings = []
    if dq.get("duplicate_timestamps_handled") and dq.get("duplicate_timestamps_count", 0) > 0:
        dq_warnings.append(f"Transaction-level duplicate timestamps ({dq['duplicate_timestamps_count']}) aggregated to forecast frequency")
    elif dq.get("duplicate_timestamps_handled"):
        dq_warnings.append("Transaction timestamps aggregated to regular forecast frequency")

    if dq.get("duplicate_rows_removed", 0) > 0:
        dq_warnings.append(f"{dq['duplicate_rows_removed']} exact duplicate rows cleaned")

    if dq.get("missing_values_handled", 0) > 0:
        dq_warnings.append(f"{dq['missing_values_handled']} missing values imputed")

    dq_warnings.append(f"Excluded identifiers/PII: {', '.join(dq.get('excluded_features', []))}")

    return {
        "selected_model": context.get("selected_model", "Random Forest"),
        "selection_reason": context.get("model_selection_reason", "Best validation performance"),
        "performance_summary": {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "r2": r2_val,
            "r2_description": r2_desc,
            "baseline_improvement_pct": baseline.get("model_improvement_pct"),
            "naive_rmse": baseline.get("naive_rmse")
        },
        "important_drivers": drivers[:5],
        "data_quality_warnings": dq_warnings,
        "forecast_reliability": [
            "Held-out temporal test set validation",
            "Model predictions bound by historical feature distributions",
            "High MAPE sensitivity warning enabled" if context.get("target_scale_warning") else "Normal scale error tracking"
        ]
    }


def generate_explanation(context: dict[str, Any]) -> dict[str, Any]:
    """Central routing function for LLM explanation generation.
    
    Order:
    1. Groq / openai/gpt-oss-120b
    2. Ollama / qwen3:8b
    3. Deterministic Local Generator
    """
    dataset_id = context.get("dataset_id", "default_dataset")
    dataset_hash = context.get("dataset_hash", "default_hash")
    
    groq_key = _get_config_var("GROQ_API_KEY")
    groq_model = _get_config_var("GROQ_MODEL", "openai/gpt-oss-120b")
    ollama_url = _get_config_var("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = _get_config_var("OLLAMA_MODEL", "qwen3:8b")
    
    sys_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(context)
    structured_summary = build_structured_summary(context)
    
    # Provider 1: Groq Primary
    if groq_key and groq_key.strip():
        groq_text = call_groq_api(groq_key.strip(), groq_model, sys_prompt, user_prompt)
        if groq_text:
            return {
                "provider": "groq",
                "model": groq_model,
                "fallback_used": False,
                "explanation": groq_text,
                "status": "success",
                "structured_summary": structured_summary,
                "dataset_id": dataset_id,
                "dataset_hash": dataset_hash,
            }

            
    # Provider 2: Ollama Fallback
    if check_ollama_model_available(ollama_url, ollama_model):
        ollama_text = call_ollama_api(ollama_url, ollama_model, sys_prompt, user_prompt)
        if ollama_text:
            return {
                "provider": "ollama",
                "model": ollama_model,
                "fallback_used": True,
                "explanation": ollama_text,
                "status": "success",
                "structured_summary": structured_summary,
                "dataset_id": dataset_id,
                "dataset_hash": dataset_hash,
            }
            
    # Provider 3: Deterministic Fallback
    det_text = generate_deterministic_explanation(context)
    return {
        "provider": "deterministic",
        "model": None,
        "fallback_used": True,
        "explanation": det_text,
        "status": "fallback",
        "structured_summary": structured_summary,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
    }


def semantic_match_columns_llm(
    columns: list[str],
    concepts: list[str],
    sample_data: dict[str, list[Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Use LLM routing (Groq -> Ollama -> Deterministic) to semantically match dataset columns to required concepts.
    
    Guarantees:
    - Never invents non-existent columns (strictly constrained to input `columns`).
    - Returns confidence score for each mapped concept.
    """
    import json
    
    valid_cols_set = set(columns)
    if not columns or not concepts:
        return {}
        
    groq_key = _get_config_var("GROQ_API_KEY")
    groq_model = _get_config_var("GROQ_MODEL", "openai/gpt-oss-120b")
    ollama_url = _get_config_var("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = _get_config_var("OLLAMA_MODEL", "qwen3:8b")
    
    sys_prompt = (
        "You are an expert data engineer for time-series forecasting. "
        "Your task is to map required semantic concepts to the EXACT column names provided in the dataset schema. "
        "RULES:\n"
        "1. You MUST only select column names that appear in the provided 'AVAILABLE COLUMNS' list.\n"
        "2. Do NOT invent, hallucinate, or alter column names.\n"
        "3. Output ONLY a valid JSON object mapping each concept to {\"column\": \"<exact_column_name>\", \"confidence\": <float_0_to_1>}.\n"
        "4. If a concept has no matching column in the dataset, set column to null and confidence to 0.0."
    )
    
    user_prompt = (
        f"AVAILABLE COLUMNS IN DATASET:\n{json.dumps(columns)}\n\n"
        f"REQUIRED CONCEPTS TO MAP:\n{json.dumps(concepts)}\n\n"
    )
    if sample_data:
        user_prompt += f"SAMPLE PREVIEW DATA (first few rows):\n{json.dumps(sample_data)}\n\n"
        
    user_prompt += "Return JSON mapping matching the concepts:"
    
    raw_response = None
    if groq_key and groq_key.strip():
        raw_response = call_groq_api(groq_key.strip(), groq_model, sys_prompt, user_prompt)
        
    if not raw_response and check_ollama_model_available(ollama_url, ollama_model):
        raw_response = call_ollama_api(ollama_url, ollama_model, sys_prompt, user_prompt)
        
    results: dict[str, dict[str, Any]] = {}
    if raw_response:
        try:
            # Extract JSON block
            json_match = re.search(r"\{[\s\S]*\}", raw_response)
            if json_match:
                parsed = json.loads(json_match.group(0))
                for concept, mapping in parsed.items():
                    if isinstance(mapping, dict):
                        col = mapping.get("column")
                        conf = float(mapping.get("confidence", 0.0))
                        if col in valid_cols_set and conf >= 0.5:
                            results[concept] = {
                                "column": col,
                                "confidence": min(1.0, conf),
                                "method": "llm_semantic",
                            }
        except Exception as err:
            logger.warning(f"Failed to parse LLM semantic match response: {err}")
            
    return results

