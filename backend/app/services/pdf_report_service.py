"""SmartXSense Professional PDF Report Generator.

Generates an authoritative, publication-quality PDF audit report for any
forecast run (held-out test evaluation or future scenario projection).
Uses ReportLab Platypus layout with clean typography, data tables, and
methodology disclosures.
"""

from __future__ import annotations

import html
import io
import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ml_engine.history.forecast_history import ForecastHistoryError, get_forecast_run
from .app_summary_service import get_report


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page count and footer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Top rule for footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 34, letter[0] - 36, 34)

        # Footer labels
        self.drawString(36, 22, "SmartXSense Academic Forecast Audit -- Leakage-Safe Chronological Split")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 22, page_str)
        self.restoreState()


def _normalize_pdf_text(text: Any) -> str:
    """Normalize Unicode characters to ReportLab Type 1 (Helvetica WinAnsi) safe equivalents.

    Eliminates black square / box artifacts (such as non-breaking hyphens U+2011, figure dashes U+2012,
    horizontal bars U+2015, zero-width spaces U+200B-U+200F, etc.) while preserving readability,
    standard punctuation, and meaningful mathematical symbols.
    """
    if text is None:
        return ""
    s = str(text)

    # 1. Hyphen and dash normalization -> standard hyphen '-'
    # U+2010 (hyphen), U+2011 (non-breaking hyphen), U+2012 (figure dash),
    # U+2015 (horizontal bar), U+2212 (minus sign), U+FE63 (small hyphen), U+FF0D (fullwidth hyphen)
    s = re.sub(r'[\u2010\u2011\u2012\u2015\u2212\ufe63\uff0d]', '-', s)

    # En dash (U+2013) -> '-'
    s = s.replace('\u2013', '-')
    # Em dash (U+2014) -> ' -- '
    s = s.replace('\u2014', ' -- ')
    # Replacement character U+FFFD -> '-'
    s = s.replace('\ufffd', '-')

    # 2. Spaces and zero-width characters:
    # Non-breaking space U+00A0, narrow no-break space U+202F, thin space U+2009, hair space U+200A,
    # en space U+2002, em space U+2003, four-per-em U+2004, math space U+205F, ideographic U+3000 -> regular space
    s = re.sub(r'[\u00a0\u2000-\u200a\u202f\u205f\u3000]', ' ', s)
    # Zero-width spaces, joiners, direction marks: U+200B through U+200F, U+FEFF (BOM), U+00AD (soft hyphen) -> empty
    s = re.sub(r'[\u200b-\u200f\ufeff\u00ad]', '', s)

    # 3. Quotes and apostrophes:
    # Single curly quotes U+2018, U+2019, U+201A, U+201B, prime U+2032 -> '
    s = re.sub(r'[\u2018\u2019\u201a\u201b\u2032]', "'", s)
    # Double curly quotes U+201C, U+201D, U+201E, U+201F, double prime U+2033, guillemets U+00AB, U+00BB -> "
    s = re.sub(r'[\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb]', '"', s)
    # Single guillemets U+2039, U+203A -> '
    s = re.sub(r'[\u2039\u203a]', "'", s)

    # 4. Ellipsis:
    # Horizontal ellipsis U+2026 -> '...'
    s = s.replace('\u2026', '...')

    # 5. Bullets and squares:
    # Black square U+25A0, small squares U+25AA, U+25AB, black circle U+25CF, white circle U+25CB -> standard bullet
    s = re.sub(r'[\u25cf\u25cb\u25aa\u25ab\u25a0]', '&bull;', s)

    # 6. Arrows:
    # U+2192 -> '->', U+2190 -> '<-', U+2194 -> '<->'
    s = s.replace('\u2192', '->').replace('\u2190', '<-').replace('\u2194', '<->')

    # 7. Math symbols:
    # Less than equal U+2264, greater than equal U+2265, not equal U+2260, approx U+2248, plus-minus U+00B1
    s = s.replace('\u2264', '<=').replace('\u2265', '>=').replace('\u2260', '!=').replace('\u2248', '~').replace('\u00b1', '+/-')
    s = s.replace('\u00d7', 'x').replace('\u00f7', '/')

    # 8. Emojis and miscellaneous symbols:
    # Strip emojis outside basic multilingual plane or in symbol blocks
    s = re.sub(r'[\U00010000-\U0010ffff]', '', s)
    s = re.sub(r'[\u2600-\u27bf]', '', s)

    return s


def _escape(text: Any) -> str:
    if text is None:
        return "-"
    return html.escape(_normalize_pdf_text(str(text)))


def _val_or_na(val: Any, default: str = "Not available", suffix: str = "") -> str:
    """Format string or return default ('Not available' / 'N/A'). Never displays zero when unknown."""
    if val is None or val == "":
        return default
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return default
    return f"{_escape(s)}{suffix}"


def _format_num(val: Any, decimals: int = 2, default: str = "-") -> str:
    if val is None or val == "":
        return default
    try:
        num = float(val)
        return f"{num:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _format_int(val: Any, default: str = "Not available") -> str:
    if val is None or val == "":
        return default
    try:
        num = int(float(val))
        return f"{num:,}"
    except (ValueError, TypeError):
        return default


def _format_pct(val: Any, default: str = "Not available") -> str:
    if val is None or val == "":
        return default
    try:
        num = float(val)
        pct = num * 100.0 if num <= 1.0 else num
        return f"{pct:.1f}%"
    except (ValueError, TypeError):
        return default


def _format_mape(val: Any) -> str:
    if val is None or val == "":
        return "-"
    try:
        num = float(val)
        pct = num * 100.0 if num <= 1.0 else num
        return f"{pct:.2f}%"
    except (ValueError, TypeError):
        return str(val)


def _scrub_internal_identifiers(text: Any) -> str:
    """Scrub internal LLM provider/model identifiers (e.g., openai/gpt-oss-120b, qwen3:8b) from user-facing text."""
    if not text:
        return ""
    s = str(text)
    s = re.sub(r'openai/gpt-oss-120b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'gpt-oss-120b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'qwen3:8b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'openai/[a-zA-Z0-9_\-\.:]+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(?:provider|model):\s*(?:groq|ollama|deterministic|[a-zA-Z0-9_\-\./]+)\b', '', s, flags=re.IGNORECASE)
    return s.strip()


def _format_inline_markdown(text: str) -> str:
    """Transform markdown bold (**), italic (*), and backticks (`) to valid ReportLab XML tags,
    escaping untrusted characters first to prevent parser errors."""
    if not text:
        return ""
    normalized = _normalize_pdf_text(text)
    cleaned = _scrub_internal_identifiers(normalized)
    # Remove emoji symbols that can cause box artifacts in standard Type 1 fonts
    cleaned = re.sub(r'[🧠⚠️🔎💡🛡️🤖📌🎯📊📈📉]', '', cleaned)
    # Normalize newline escapes and excessive spacing
    cleaned = cleaned.replace("\\r\\n", " ").replace("\\n", " ").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Escape HTML special characters
    escaped = html.escape(cleaned)

    # Convert ***bold italic***
    escaped = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', escaped)
    # Convert **bold**
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    # Convert *italic* (non-whitespace bounded)
    escaped = re.sub(r'(?<!\w)\*([^\*\s]+.*?)\*(?!\w)', r'<i>\1</i>', escaped)
    # Convert `code` (feature names in distinct bold primary color)
    escaped = re.sub(r'`([^`]+)`', r'<b><font color="#0284C7">\1</font></b>', escaped)
    # Clean any dangling asterisks, hashes, or quotes
    escaped = escaped.replace("**", "").replace("###", "")
    return escaped.strip()


def _parse_explanation_content(narrative: str) -> dict[str, list[str]]:
    """Parse raw explanation narrative into clean structured sections.
    Handles standard markdown headings, malformed newline headings (### n...), bold headings,
    and unstructured free-form text without losing content or leaking markdown tokens."""
    if not narrative:
        return {}

    cleaned = _normalize_pdf_text(narrative)
    cleaned = _scrub_internal_identifiers(cleaned)
    # Normalize newline variations
    cleaned = cleaned.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # Fix malformed newline characters adjacent to ### (e.g. '### n ...' or '### nn ...')
    cleaned = re.sub(r'###\s*n+\s+', '### ', cleaned)
    cleaned = re.sub(r'###\s*\n+\s*', '### ', cleaned)

    lines = cleaned.split("\n")
    sections: dict[str, list[str]] = {
        "summary": [],
        "situation": [],
        "factors": [],
        "actions": [],
        "precautions": [],
    }

    current_section = "summary"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for section heading
        # 1. ### Heading
        h_match = re.match(r'^#{1,4}\s*(.*)$', stripped)
        # 2. **Heading**:
        b_match = not h_match and re.match(r'^\s*(?:\*\*)?([^:*]+?)(?:\*\*)?:?\s*$', stripped)

        heading_candidate = ""
        if h_match:
            heading_candidate = h_match.group(1).strip()
        elif b_match:
            cand = b_match.group(1).strip()
            if any(k in cand.lower() for k in ("summary", "situation", "why", "causes", "factors", "actions", "remedies", "precautions", "monitoring")):
                heading_candidate = cand

        if heading_candidate:
            h_lower = heading_candidate.lower()
            if "summary" in h_lower:
                current_section = "summary"
                continue
            elif "situation" in h_lower:
                current_section = "situation"
                continue
            elif any(k in h_lower for k in ("why", "cause", "factor", "driver", "influenc")):
                current_section = "factors"
                continue
            elif any(k in h_lower for k in ("action", "remed", "recommend", "guidance")):
                current_section = "actions"
                continue
            elif any(k in h_lower for k in ("precaution", "monitor", "safeguard")):
                current_section = "precautions"
                continue

        # Not a heading line: add content to current section
        bullet_match = re.match(r'^[\s*•\-–—]+(?:\d+\.)?\s*(.*)$', stripped)
        content_text = bullet_match.group(1).strip() if bullet_match and (current_section in ("actions", "precautions", "factors")) else stripped

        if content_text:
            sections[current_section].append(content_text)

    return sections


def _attribution_role(feat_name: str, target: str = "") -> str:
    feat_lower = str(feat_name).lower()
    if "lag_1" in feat_lower or "lag 1" in feat_lower:
        return "Short-term temporal anchor (immediate prior period)"
    if "lag_24" in feat_lower or "hour" in feat_lower or "diurnal" in feat_lower:
        return "Diurnal cycle regulator (24-hour periodicity)"
    if "168" in feat_lower or "lag_7" in feat_lower or "lag 7" in feat_lower or "week" in feat_lower:
        return "Weekly seasonality baseline (7-day periodicity)"
    if "ewma" in feat_lower or "exp_moving" in feat_lower:
        return "Recent exponentially smoothed demand momentum"
    if "rolling" in feat_lower or "moving_avg" in feat_lower:
        return "Trend smoothing moving average window"
    if "month" in feat_lower:
        return "Macro seasonal month-of-year periodic contribution"
    if "day_of_week" in feat_lower or "dayofweek" in feat_lower or "weekday" in feat_lower:
        return "Weekday vs. weekend intra-week demand variation"
    if "inflation" in feat_lower or "cpi" in feat_lower:
        return "Macroeconomic purchasing-power and price-level index"
    if any(w in feat_lower for w in ("temp", "weather", "rain", "humidity")):
        return "Exogenous environmental / weather context"
    if any(p in feat_lower for p in ("price", "discount", "promo", "fuel")):
        return "Commercial pricing / promotional elasticity driver"
    target_clean = target if target and target != "target" else "demand target"
    return f"Contextual predictor for {target_clean}"


def _extract_dataset_intelligence(report: dict[str, Any]) -> dict[str, Any]:
    """Robustly extract Dataset Intelligence facts from report payloads or nested artifacts."""
    ds_intel = report.get("dataset_intelligence")
    if not isinstance(ds_intel, dict):
        ds_intel = (report.get("dataset_info") or {}).get("intelligence")
    if not isinstance(ds_intel, dict):
        ds_intel = {}

    ds_info = report.get("dataset_info") or {}
    ds_ctx = report.get("dataset_context") or {}
    clean_sum = ds_ctx.get("cleaning_summary") or {}
    suit_data = ds_ctx.get("suitability") or {}

    ds_name = (
        ds_intel.get("dataset_name")
        or ds_info.get("dataset_name")
        or ds_ctx.get("dataset_name")
        or ds_info.get("dataset_id")
        or "SmartXSense Dataset"
    )

    orig_rows = (
        ds_intel.get("original_row_count")
        or clean_sum.get("raw_row_count")
        or ds_ctx.get("raw_rows")
        or ds_ctx.get("rows_used")
        or ds_info.get("observation_count")
    )
    orig_cols = (
        ds_intel.get("original_column_count")
        or ds_ctx.get("column_count")
    )
    cleaned_rows = (
        ds_intel.get("cleaned_row_count")
        or clean_sum.get("final_cleaned_rows")
        or ds_ctx.get("cleaned_rows")
    )

    tr_obs = ds_intel.get("training_observations") or ds_ctx.get("train_rows")
    val_obs = ds_intel.get("validation_observations") or ds_ctx.get("validation_rows")
    te_obs = ds_intel.get("test_observations") or ds_ctx.get("test_rows")
    f_horizon = (
        ds_intel.get("forecast_horizon")
        or report.get("horizon")
        or report.get("prediction_count")
        or (len(report.get("forecast_data", {}).get("points", [])) if report.get("forecast_data") else None)
    )

    time_col = (
        ds_intel.get("detected_time_column")
        or ds_info.get("time_column")
        or ds_ctx.get("time_column")
        or "timestamp"
    )
    target_col = (
        ds_intel.get("detected_target_column")
        or report.get("target")
        or ds_info.get("target")
        or ds_ctx.get("target")
        or "target"
    )
    canonical_target = ds_intel.get("canonical_target") or report.get("canonical_target") or target_col

    unique_ts = ds_intel.get("unique_timestamps") or ds_ctx.get("unique_timestamps") or suit_data.get("unique_timestamps")
    dup_rows = (
        ds_intel.get("duplicate_rows_removed")
        if ds_intel.get("duplicate_rows_removed") is not None
        else (clean_sum.get("exact_duplicates_removed") if clean_sum.get("exact_duplicates_removed") is not None else ds_ctx.get("duplicate_rows_removed"))
    )
    dup_ts = (
        ds_intel.get("duplicate_timestamps_count")
        if ds_intel.get("duplicate_timestamps_count") is not None
        else (clean_sum.get("duplicate_timestamps_count") if clean_sum.get("duplicate_timestamps_count") is not None else ds_ctx.get("duplicate_timestamps_count"))
    )
    missing_imputed = (
        ds_intel.get("missing_values_handled")
        if ds_intel.get("missing_values_handled") is not None
        else (clean_sum.get("missing_values_handled") if clean_sum.get("missing_values_handled") is not None else ds_ctx.get("missing_values_handled"))
    )

    excl_feats = (
        ds_intel.get("excluded_features")
        or clean_sum.get("excluded_features")
        or ds_ctx.get("excluded_features")
        or []
    )
    excl_reasons = (
        ds_intel.get("excluded_reasons")
        or clean_sum.get("excluded_reasons")
        or ds_ctx.get("excluded_reasons")
        or {}
    )
    num_forecast_features = (
        ds_intel.get("final_forecasting_features_count")
        or ds_ctx.get("final_forecasting_features_count")
    )

    fit_score = ds_intel.get("domain_fit_score") or ds_ctx.get("confidence") or suit_data.get("confidence")
    fit_label = ds_intel.get("domain_fit_label") or ds_ctx.get("confidence_label") or suit_data.get("confidence_label")

    suit_status = (
        ds_intel.get("suitability_status")
        or suit_data.get("classification")
        or ("SUITABLE" if (ds_intel.get("suitable") or suit_data.get("is_suitable")) else None)
    )
    suit_label = (
        ds_intel.get("suitability_label")
        or suit_data.get("status_label")
        or ("✓ Fully Suitable" if suit_status == "SUITABLE" else ("⚠️ Suitable with Warnings" if suit_status == "SUITABLE_WITH_WARNINGS" else "⚠️ Unsuitable"))
        if suit_status
        else None
    )

    suit_warnings = ds_intel.get("suitability_warnings") or suit_data.get("warnings") or []
    suit_reasons = ds_intel.get("suitability_reasons") or suit_data.get("reasons") or []
    explanation = (
        ds_intel.get("explanation")
        or suit_data.get("explanation")
        or f"Dataset inspected and verified for {report.get('domain', 'forecasting').capitalize()} time-series forecasting."
    )

    return {
        "dataset_name": ds_name,
        "original_row_count": orig_rows,
        "original_column_count": orig_cols,
        "cleaned_row_count": cleaned_rows,
        "training_observations": tr_obs,
        "validation_observations": val_obs,
        "test_observations": te_obs,
        "forecast_horizon": f_horizon,
        "detected_time_column": time_col,
        "detected_target_column": target_col,
        "canonical_target": canonical_target,
        "unique_timestamps": unique_ts,
        "duplicate_rows_removed": dup_rows,
        "duplicate_timestamps_count": dup_ts,
        "missing_values_handled": missing_imputed,
        "excluded_features": excl_feats,
        "excluded_reasons": excl_reasons,
        "final_forecasting_features_count": num_forecast_features,
        "domain_fit_score": fit_score,
        "domain_fit_label": fit_label,
        "suitability_status": suit_status,
        "suitability_label": suit_label,
        "suitability_warnings": suit_warnings,
        "suitability_reasons": suit_reasons,
        "explanation": explanation,
    }


def _extract_explainability(report: dict[str, Any]) -> dict[str, Any]:
    """Safely extract SHAP, LIME, native feature importances, and validation metrics."""
    exp_obj = report.get("explainability") or {}
    model_name = str(report.get("model") or "").replace("_", " ").title()
    raw_model = str(report.get("model") or "").lower().strip()
    val_bench = report.get("validation_benchmark") or {}
    model_info = report.get("model_info") or {}

    val_rmse = exp_obj.get("validation_rmse")
    if val_rmse is None and isinstance(val_bench, dict) and val_bench.get("ranking"):
        for cand in val_bench["ranking"]:
            if cand.get("model") == raw_model or cand.get("rank") == 1:
                val_rmse = (cand.get("validation_metrics") or cand.get("metrics") or {}).get("rmse")
                if val_rmse is not None:
                    break
    if val_rmse is None and isinstance(report.get("validation_metrics"), dict):
        val_rmse = report["validation_metrics"].get("rmse")

    selection_reason = (
        exp_obj.get("selection_reason")
        or report.get("selection_reason")
        or (f"Achieved lowest Validation RMSE ({_format_num(val_rmse)}) among evaluated candidates" if val_rmse is not None else "User specified model")
    )
    selection_criterion = (
        exp_obj.get("selection_criterion")
        or model_info.get("selection_criterion")
        or "Lowest Validation RMSE"
    )

    feat_imp = (
        exp_obj.get("feature_importance")
        or report.get("feature_importance")
        or report.get("feature_importances")
    )
    shap_data = (
        exp_obj.get("shap_features")
        or exp_obj.get("shap_explanation")
        or report.get("shap_explanation")
        or report.get("shap_features")
    )
    lime_data = exp_obj.get("lime_explanation") or report.get("lime_explanation")
    llm_exp = exp_obj.get("ai_explanation") or report.get("llm_explanation")

    return {
        "selected_model": model_name,
        "raw_model": raw_model,
        "selection_reason": selection_reason,
        "selection_criterion": selection_criterion,
        "validation_rmse": val_rmse,
        "feature_importance": feat_imp,
        "shap_data": shap_data,
        "lime_data": lime_data,
        "ai_explanation": llm_exp,
    }


def build_pdf_document(report: dict[str, Any]) -> io.BytesIO:
    """Compile structured report metadata into an in-memory PDF buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0284C7"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    h3_style = ParagraphStyle(
        "Heading3",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
    xai_subheading_style = ParagraphStyle(
        "XAISubHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#0284C7"),
        spaceBefore=6,
        spaceAfter=2.5,
        keepWithNext=True,
    )
    xai_body_style = ParagraphStyle(
        "XAIBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3,
    )
    xai_bullet_style = ParagraphStyle(
        "XAIBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2,
    )
    xai_disclosure_style = ParagraphStyle(
        "XAIDisclosure",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#64748B"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    body_muted = ParagraphStyle(
        "BodyMuted",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )
    small_bold = ParagraphStyle(
        "SmallBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
        keepWithNext=True,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#334155"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0284C7"),
    )

    story: list[Any] = []

    run_id = str(report.get("run_id") or report.get("report_id") or "UNKNOWN")
    domain = str(report.get("domain") or "").capitalize()
    model_name = str(report.get("model") or "").replace("_", " ").title()
    target = str(report.get("target") or "target")
    f_type = str(report.get("forecast_type") or "test_evaluation")
    type_label = "Held-Out Test Evaluation" if f_type == "test_evaluation" else "Future Horizon Forecast"
    timestamp_str = str(report.get("timestamp") or "—")

    # Header Title Banner
    story.append(Paragraph("SmartXSense — Forecast & System Audit Report", title_style))
    story.append(
        Paragraph(
            f"Explainable Hybrid Demand Forecasting System  |  Run ID: {_escape(run_id)}  |  Generated: {_escape(timestamp_str)}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceBefore=1, spaceAfter=8))

    # Metadata & Context Grid
    ds_info = report.get("dataset_info") or {}
    model_info = report.get("model_info") or {}
    geography = ds_info.get("geographic_scope") or "All Geographies"
    horizon_val = report.get("horizon") or report.get("prediction_count") or "—"
    artifact_source = report.get("artifact_source") or model_info.get("artifact_source") or "trained"

    xai = _extract_explainability(report)
    val_rmse_val = xai.get("validation_rmse")
    val_rmse_str = f"{_format_num(val_rmse_val)} {target}" if val_rmse_val is not None else "Not available"

    meta_rows = [
        [
            Paragraph(f"<b>Domain:</b> {_escape(domain)}", body_style),
            Paragraph(f"<b>Target Variable:</b> {_escape(target)}", body_style),
            Paragraph(f"<b>Selected Model:</b> {_escape(model_name)}", body_style),
        ],
        [
            Paragraph(f"<b>Forecast Type:</b> {_escape(type_label)}", body_style),
            Paragraph(f"<b>Geographic Scope:</b> {_escape(geography)}", body_style),
            Paragraph(f"<b>Artifact Source:</b> {_escape(artifact_source).capitalize()}", body_style),
        ],
        [
            Paragraph(f"<b>Horizon / Points:</b> {_escape(horizon_val)} observations", body_style),
            Paragraph(f"<b>Selection Criterion:</b> Lowest Validation RMSE", body_style),
            Paragraph(f"<b>Validation RMSE:</b> {_escape(val_rmse_str)}", body_style),
        ],
    ]
    meta_table = Table(meta_rows, colWidths=[180, 180, 180])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Section 1: Executive Forecast Summary & Behavioral Dynamics
    summary_obj = report.get("forecast_summary") or {}
    story.append(Paragraph("1. Executive Forecast Summary & Behavioral Dynamics", h2_style))
    summary_text = summary_obj.get("summary") or f"Forecast executed for {domain.lower()} predicting {target} using {model_name}."
    interp_text = summary_obj.get("interpretation") or ""
    full_narrative = f"{summary_text} {interp_text}".strip()
    story.append(Paragraph(_escape(full_narrative), body_style))

    insights = summary_obj.get("insights") or []
    if insights:
        story.append(Spacer(1, 3))
        for insight in insights[:4]:
            bullet_html = f"&bull; <i>{_escape(insight)}</i>"
            story.append(Paragraph(bullet_html, body_muted))
    story.append(Spacer(1, 8))

    # Section 2: Dataset Intelligence & Data Quality Audit
    intel = _extract_dataset_intelligence(report)
    story.append(Paragraph("2. Dataset Intelligence & Data Quality Audit", h2_style))
    story.append(
        Paragraph(
            "<i>Automated schema inspection, data quality audit, and domain-fit suitability analysis:</i>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 3))

    orig_rows = intel.get("original_row_count")
    orig_cols = intel.get("original_column_count")
    orig_dims_str = f"{_format_int(orig_rows)} rows × {_format_int(orig_cols)} cols" if (orig_rows is not None and orig_cols is not None) else (_format_int(orig_rows, "Not available") + " rows" if orig_rows is not None else "Not available")

    time_col = intel.get("detected_time_column")
    target_col = intel.get("detected_target_column")
    canon_target = intel.get("canonical_target")
    target_display = f"{target_col} (canonical: {canon_target})" if (target_col and canon_target and target_col != canon_target) else (target_col or target)

    fit_score = intel.get("domain_fit_score")
    fit_label = intel.get("domain_fit_label")
    if fit_score is not None:
        fit_str = f"{_format_pct(fit_score)} ({_escape(fit_label)})" if fit_label else _format_pct(fit_score)
    else:
        fit_str = "Not available"

    suit_status_label = intel.get("suitability_label") or intel.get("suitability_status") or "Not available"

    intel_table_data = [
        [
            Paragraph("<b>Dataset Identifier:</b>", small_bold),
            Paragraph(_val_or_na(intel.get("dataset_name")), table_cell),
            Paragraph("<b>Original Dimensions:</b>", small_bold),
            Paragraph(orig_dims_str, table_cell),
        ],
        [
            Paragraph("<b>Detected Date Column:</b>", small_bold),
            Paragraph(_val_or_na(time_col), table_cell),
            Paragraph("<b>Cleaned Dataset Size:</b>", small_bold),
            Paragraph(_format_int(intel.get("cleaned_row_count"), default="Not available") + (" rows" if intel.get("cleaned_row_count") is not None else ""), table_cell),
        ],
        [
            Paragraph("<b>Detected Target Column:</b>", small_bold),
            Paragraph(_escape(target_display), table_cell),
            Paragraph("<b>Forecast Features:</b>", small_bold),
            Paragraph(_format_int(intel.get("final_forecasting_features_count"), default="Not available") + (" features" if intel.get("final_forecasting_features_count") is not None else ""), table_cell),
        ],
        [
            Paragraph("<b>Unique Timestamps:</b>", small_bold),
            Paragraph(_format_int(intel.get("unique_timestamps"), default="Not available"), table_cell),
            Paragraph("<b>Domain-Fit Score:</b>", small_bold),
            Paragraph(_escape(fit_str), table_cell),
        ],
        [
            Paragraph("<b>Duplicates Removed:</b>", small_bold),
            Paragraph(_format_int(intel.get("duplicate_rows_removed"), default="Not available") + (" rows" if intel.get("duplicate_rows_removed") is not None else ""), table_cell),
            Paragraph("<b>Suitability Status:</b>", small_bold),
            Paragraph(_escape(suit_status_label), small_bold if "suitable" in suit_status_label.lower() else table_cell),
        ],
        [
            Paragraph("<b>Timestamps Handled:</b>", small_bold),
            Paragraph(_format_int(intel.get("duplicate_timestamps_count"), default="Not available") + (" handled" if intel.get("duplicate_timestamps_count") is not None else ""), table_cell),
            Paragraph("<b>Missing Values Imputed:</b>", small_bold),
            Paragraph(_format_int(intel.get("missing_values_handled"), default="Not available") + (" imputed" if intel.get("missing_values_handled") is not None else ""), table_cell),
        ],
    ]

    intel_table = Table(intel_table_data, colWidths=[130, 140, 130, 140])
    intel_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(intel_table)
    story.append(Spacer(1, 4))

    # Sub-table: Observational Split Breakdown
    story.append(Paragraph("<b>Observational Breakdown Across Time-Series Partitions:</b>", h3_style))
    split_headers = [
        Paragraph("<b>Original Dataset</b>", table_header),
        Paragraph("<b>Cleaned Dataset</b>", table_header),
        Paragraph("<b>Training Obs</b>", table_header),
        Paragraph("<b>Validation Obs</b>", table_header),
        Paragraph("<b>Held-Out Test Obs</b>", table_header),
        Paragraph("<b>Forecast Horizon</b>", table_header),
    ]
    split_values = [
        Paragraph(_format_int(intel.get("original_row_count")), table_cell_center),
        Paragraph(_format_int(intel.get("cleaned_row_count")), table_cell_center),
        Paragraph(_format_int(intel.get("training_observations")), table_cell_center),
        Paragraph(_format_int(intel.get("validation_observations")), table_cell_center),
        Paragraph(_format_int(intel.get("test_observations")), table_cell_center),
        Paragraph(f"{_escape(intel.get('forecast_horizon') or horizon_val)} steps", table_cell_center),
    ]
    split_table = Table([split_headers, split_values], colWidths=[90, 90, 90, 90, 90, 90])
    split_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F1F5F9")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(split_table)
    story.append(Spacer(1, 4))

    # Feature Governance & Excluded Columns Breakdown
    excl_cols = intel.get("excluded_features") or []
    excl_reasons = intel.get("excluded_reasons") or {}
    if excl_cols:
        story.append(Paragraph("<b>Feature Governance & Excluded Attributes:</b>", h3_style))
        gov_headers = [
            Paragraph("<b>Excluded Feature / Column</b>", table_header),
            Paragraph("<b>Governance Classification</b>", table_header),
            Paragraph("<b>Exclusion Rationale</b>", table_header),
        ]
        gov_rows = [gov_headers]
        for col_name in excl_cols[:6]:
            col_str = str(col_name)
            reason_text = excl_reasons.get(col_str) or (
                "Entity identifier / personal data code excluded to avoid memorization and leakage"
                if any(k in col_str.lower() for k in ("id", "code", "customer", "invoice", "sku"))
                else "Non-predictive contextual or redundant temporal attribute"
            )
            cat_text = "Identifier / Key" if "id" in col_str.lower() else "Personal / Entity Data"
            gov_rows.append([
                Paragraph(_escape(col_str), small_bold),
                Paragraph(_escape(cat_text), table_cell),
                Paragraph(_escape(reason_text), table_cell),
            ])
        gov_table = Table(gov_rows, colWidths=[150, 120, 270])
        gov_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFFBEB")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FDE68A")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FDE68A")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(gov_table)
        story.append(Spacer(1, 3))

    # Suitability Conclusion & Quality Outcome
    explanation_text = intel.get("explanation") or "Dataset validated for time-series forecasting."
    story.append(Paragraph(f"<b>Data Quality Outcome:</b> {_escape(explanation_text)}", body_style))
    suit_warnings = intel.get("suitability_warnings") or []
    if suit_warnings:
        story.append(Spacer(1, 2))
        warn_str = " | ".join(str(w) for w in suit_warnings)
        story.append(Paragraph(f"<b>Quality Warnings:</b> {_escape(warn_str)}", body_muted))
    story.append(Spacer(1, 8))

    # Section 3: Forecast Performance & Model Comparison
    metrics = report.get("metrics") or (report.get("test_evaluation") or {}).get("metrics") or {}
    story.append(Paragraph("3. Forecast Performance & Model Comparison", h2_style))
    if f_type == "test_evaluation" and metrics:
        story.append(
            Paragraph(
                "<i>Metrics evaluated strictly on the chronologically held-out test split (zero lookahead leakage):</i>",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 3))

        perf_headers = [
            Paragraph("<b>Test RMSE</b>", table_header),
            Paragraph("<b>Test MAE</b>", table_header),
            Paragraph("<b>Test MAPE</b>", table_header),
            Paragraph("<b>Test R²</b>", table_header),
            Paragraph("<b>Test WAPE</b>", table_header),
        ]
        perf_values = [
            Paragraph(f"{_format_num(metrics.get('rmse'))} {target}", table_cell),
            Paragraph(f"{_format_num(metrics.get('mae'))} {target}", table_cell),
            Paragraph(_format_mape(metrics.get("mape")), table_cell),
            Paragraph(_format_num(metrics.get("r2"), 3), table_cell),
            Paragraph(_format_mape(metrics.get("wape")) if metrics.get("wape") is not None else "—", table_cell),
        ]
        perf_table = Table([perf_headers, perf_values], colWidths=[108, 108, 108, 108, 108])
        perf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F1F5F9")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(perf_table)
    else:
        # Future forecast performance summary
        f_summary = (report.get("forecast_data") or {}).get("summary") or {}
        story.append(
            Paragraph(
                "<i>Future out-of-sample forecast. Ground truth actual values do not exist for future timestamps.</i>",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 3))
        fut_headers = [
            Paragraph("<b>Horizon Steps</b>", table_header),
            Paragraph("<b>Projected Mean</b>", table_header),
            Paragraph("<b>Projected Minimum</b>", table_header),
            Paragraph("<b>Projected Maximum</b>", table_header),
            Paragraph("<b>Scenario Direction</b>", table_header),
        ]
        fut_values = [
            Paragraph(f"{_escape(report.get('horizon') or f_summary.get('count') or '—')} steps", table_cell),
            Paragraph(f"{_format_num(f_summary.get('mean'))} {target}", table_cell),
            Paragraph(f"{_format_num(f_summary.get('min'))} {target}", table_cell),
            Paragraph(f"{_format_num(f_summary.get('max'))} {target}", table_cell),
            Paragraph(_escape((report.get("summary_facts") or {}).get("forecast_direction", "Stable")).capitalize(), table_cell),
        ]
        fut_table = Table([fut_headers, fut_values], colWidths=[108, 108, 108, 108, 108])
        fut_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F1F5F9")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(fut_table)

    story.append(Spacer(1, 5))

    # Model Selection & Benchmark Table
    val_bench = report.get("validation_benchmark") or {}
    ranking = val_bench.get("ranking") or []
    story.append(
        Paragraph(
            "<b>Validation Benchmark & Selection Ranking:</b> "
            "<i>(Auto-selection strictly on chronologically separated validation set. Lowest Validation RMSE wins)</i>",
            body_muted,
        )
    )
    story.append(Spacer(1, 2))

    if ranking:
        bench_rows = [
            [
                Paragraph("<b>Rank</b>", table_header),
                Paragraph("<b>Candidate Model</b>", table_header),
                Paragraph("<b>Validation RMSE</b>", table_header),
                Paragraph("<b>Validation MAE</b>", table_header),
                Paragraph("<b>Validation MAPE</b>", table_header),
                Paragraph("<b>Selection Status</b>", table_header),
            ]
        ]
        for idx, item in enumerate(ranking[:6], start=1):
            m_key = item.get("model", "")
            m_label = m_key.replace("_", " ").title()
            v_metrics = item.get("validation_metrics") or item.get("metrics") or {}
            is_winner = (idx == 1) or (m_key.lower() == model_name.lower().replace(" ", "_"))
            status_text = "<b>Selected (Winner)</b>" if is_winner else "Evaluated"

            bench_rows.append([
                Paragraph(str(idx), table_cell),
                Paragraph(_escape(m_label), small_bold if is_winner else table_cell),
                Paragraph(_format_num(v_metrics.get("rmse")), table_cell),
                Paragraph(_format_num(v_metrics.get("mae")), table_cell),
                Paragraph(_format_mape(v_metrics.get("mape")), table_cell),
                Paragraph(status_text, small_bold if is_winner else table_cell),
            ])

        bench_table = Table(bench_rows, colWidths=[36, 170, 84, 84, 84, 82])
        bench_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E0F2FE")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BAE6FD")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(bench_table)
    else:
        story.append(
            Paragraph(
                f"Model <b>{_escape(model_name)}</b> was selected via {report.get('selection_method', 'manual selection')}. "
                f"Selection criterion: {model_info.get('selection_criterion', 'Lowest Validation RMSE')}.",
                body_style,
            )
        )
    story.append(Spacer(1, 8))

    # Section 4: Complete Forecast Observations
    f_data = report.get("forecast_data") or {}
    points = f_data.get("points") or []
    if points:
        story.append(Paragraph("4. Complete Forecast Observations", h2_style))
        story.append(
            Paragraph(
                "<i>Chronological time-series forecast observations with numeric precision:</i>",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 3))

        # Check for optional confidence bounds
        has_ci = any(pt.get("lower_bound") is not None or pt.get("lower") is not None for pt in points) and any(
            pt.get("upper_bound") is not None or pt.get("upper") is not None for pt in points
        )

        if f_type == "test_evaluation":
            if has_ci:
                f_headers = [
                    Paragraph("<b>Timestamp (UTC)</b>", table_header),
                    Paragraph("<b>Predicted</b>", table_header),
                    Paragraph("<b>Lower (95%)</b>", table_header),
                    Paragraph("<b>Upper (95%)</b>", table_header),
                    Paragraph("<b>Actual (Truth)</b>", table_header),
                    Paragraph("<b>Residual</b>", table_header),
                ]
                col_widths = [130, 80, 80, 80, 85, 85]
            else:
                f_headers = [
                    Paragraph("<b>Timestamp (UTC)</b>", table_header),
                    Paragraph("<b>Predicted</b>", table_header),
                    Paragraph("<b>Actual (Ground Truth)</b>", table_header),
                    Paragraph("<b>Residual (Error)</b>", table_header),
                ]
                col_widths = [150, 130, 130, 130]
        else:
            if has_ci:
                f_headers = [
                    Paragraph("<b>Timestamp (UTC)</b>", table_header),
                    Paragraph("<b>Predicted</b>", table_header),
                    Paragraph("<b>Lower Bound (95%)</b>", table_header),
                    Paragraph("<b>Upper Bound (95%)</b>", table_header),
                ]
                col_widths = [180, 120, 120, 120]
            else:
                f_headers = [
                    Paragraph("<b>Timestamp (UTC)</b>", table_header),
                    Paragraph("<b>Predicted</b>", table_header),
                ]
                col_widths = [270, 270]

        MAX_PDF_ROWS = 500
        display_pts = points[:MAX_PDF_ROWS]

        pt_rows = [f_headers]
        for pt in display_pts:
            ts_label = str(pt.get("timestamp", ""))[:19].replace("T", " ")
            p_val = _format_num(pt.get("predicted"))
            row = [Paragraph(ts_label, table_cell), Paragraph(p_val, table_cell)]

            if has_ci:
                low_val = _format_num(pt.get("lower_bound") if pt.get("lower_bound") is not None else pt.get("lower"))
                upp_val = _format_num(pt.get("upper_bound") if pt.get("upper_bound") is not None else pt.get("upper"))
                row.extend([Paragraph(low_val, table_cell), Paragraph(upp_val, table_cell)])

            if f_type == "test_evaluation":
                a_val = _format_num(pt.get("actual"))
                r_val = _format_num(pt.get("residual"))
                row.extend([Paragraph(a_val, table_cell), Paragraph(r_val, table_cell)])

            pt_rows.append(row)

        obs_table = Table(pt_rows, colWidths=col_widths, repeatRows=1)
        table_style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        for r_idx in range(1, len(pt_rows)):
            bg = colors.HexColor("#F8FAFC") if (r_idx % 2 == 0) else colors.white
            table_style_cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), bg))
        obs_table.setStyle(TableStyle(table_style_cmds))
        story.append(obs_table)

        story.append(Spacer(1, 2))
        if len(points) > len(display_pts):
            story.append(
                Paragraph(
                    f"<i>Note: Displaying first {len(display_pts)} of {len(points)} forecast observations "
                    f"(compact table layout; truncated at safety threshold of {MAX_PDF_ROWS} rows for PDF rendering efficiency). "
                    f"Full data export available in CSV and JSON formats.</i>",
                    body_muted,
                )
            )
        else:
            story.append(
                Paragraph(
                    f"<i>Displaying all {len(points)} forecast observations in chronological sequence. "
                    f"Full complete dataset also available in CSV and JSON formats.</i>",
                    body_muted,
                )
            )
        story.append(Spacer(1, 8))

    # Section 5: Explainable AI (XAI) Summary
    story.append(Paragraph("5. Explainable AI (XAI) Summary", h2_style))
    story.append(
        Paragraph(
            "<i>Model interpretability, global feature attributions, and local prediction explanations:</i>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 3))

    # Subsection 5.1: Model Selection & Rationale
    xai_meta = [
        [
            Paragraph("<b>Selected Model:</b>", small_bold),
            Paragraph(_escape(_scrub_internal_identifiers(xai["selected_model"])), table_cell),
            Paragraph("<b>Selection Criterion:</b>", small_bold),
            Paragraph(_escape(xai["selection_criterion"]), table_cell),
        ],
        [
            Paragraph("<b>Selection Reason:</b>", small_bold),
            Paragraph(_escape(_scrub_internal_identifiers(xai["selection_reason"])), table_cell),
            Paragraph("<b>Validation RMSE:</b>", small_bold),
            Paragraph(_escape(val_rmse_str), table_cell),
        ],
    ]
    xai_meta_table = Table(xai_meta, colWidths=[130, 140, 130, 140])
    xai_meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(xai_meta_table)
    story.append(Spacer(1, 5))

    # Parse narrative content from ai_explanation
    ai_exp = xai.get("ai_explanation")
    narrative_found = ""
    if isinstance(ai_exp, dict):
        narrative_found = ai_exp.get("explanation") or ai_exp.get("summary") or ""
    elif isinstance(ai_exp, str) and ai_exp.strip():
        narrative_found = ai_exp.strip()

    parsed = _parse_explanation_content(narrative_found)

    # Subsection 5.2: AI Forecast Summary
    story.append(Paragraph("<b>AI Forecast Summary</b>", xai_subheading_style))
    summary_paras = parsed.get("summary") or []
    if summary_paras:
        for p in summary_paras:
            story.append(Paragraph(_format_inline_markdown(p), xai_body_style))
    else:
        t_label = (report.get("summary_facts") or {}).get("forecast_direction", "stable").lower()
        if f_type == "test_evaluation" and metrics:
            summary_synth = (
                f"Evaluation forecast for {domain.lower()} projecting {target} using {model_name}. "
                f"On the held-out test split, the model achieved an RMSE of {_format_num(metrics.get('rmse'))} "
                f"and an MAE of {_format_num(metrics.get('mae'))}, explaining {_format_pct(metrics.get('r2'))} of variance."
            )
        else:
            summary_synth = (
                f"Future out-of-sample forecast for {domain.lower()} projecting {target} using {model_name}. "
                f"Quantitative held-out error metrics are not applicable for future scenario horizons as future observations "
                f"have not yet occurred. Model trajectory indicates a {t_label} trend based on historical autoregressive dependencies."
            )
        story.append(Paragraph(_format_inline_markdown(summary_synth), xai_body_style))

    if f_type == "future" and not any("quantitative" in p.lower() for p in summary_paras):
        future_metric_notice = (
            "<i>Note: Quantitative held-out error metrics (RMSE/MAE) are not available for future scenario horizons "
            "because ground-truth observations do not exist for future timestamps.</i>"
        )
        story.append(Paragraph(future_metric_notice, body_muted))

    story.append(Spacer(1, 4))

    # Subsection 5.3: Situation Analysis
    story.append(Paragraph("<b>Situation Analysis</b>", xai_subheading_style))
    situation_paras = parsed.get("situation") or []
    if situation_paras:
        for p in situation_paras:
            story.append(Paragraph(_format_inline_markdown(p), xai_body_style))
    else:
        t_label = (report.get("summary_facts") or {}).get("forecast_direction", "stable").lower()
        sit_synth = (
            f"The calculated forecast projects a {t_label} trajectory in {target} demand across the {horizon_val}-observation horizon. "
            f"Periodic fluctuations and cyclical recurring patterns characterize the upcoming horizon without exceeding historical dispersion bounds."
        )
        story.append(Paragraph(_format_inline_markdown(sit_synth), xai_body_style))
    story.append(Spacer(1, 4))

    # Subsection 5.4: Key Influencing Factors
    story.append(Paragraph("<b>Key Influencing Factors</b>", xai_subheading_style))
    story.append(
        Paragraph(
            "<i>Global feature attributions and predictive contributions derived from the model:</i>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 2))

    # Determine whether SHAP or Native is available
    shap_data = xai.get("shap_data")
    feat_imp = xai.get("feature_importance")

    sorted_features: list[tuple[str, float]] = []
    explanation_method_label = ""
    is_shap = False

    if isinstance(shap_data, list) and shap_data:
        is_shap = True
        explanation_method_label = "Tree SHAP (Shapley Additive Explanations) -- Mean Absolute Attribution Magnitude"
        for item in shap_data:
            if isinstance(item, dict):
                f_n = item.get("feature") or item.get("name")
                f_v = item.get("importance") or item.get("shap_value") or item.get("value", 0.0)
                try:
                    sorted_features.append((str(f_n), float(f_v)))
                except (ValueError, TypeError):
                    pass
    elif isinstance(shap_data, dict) and shap_data:
        is_shap = True
        explanation_method_label = "Tree SHAP (Shapley Additive Explanations) -- Mean Absolute Attribution Magnitude"
        for k, v in shap_data.items():
            try:
                sorted_features.append((str(k), float(v)))
            except (ValueError, TypeError):
                pass
    elif isinstance(feat_imp, dict) and feat_imp:
        explanation_method_label = "Native Model Feature Importance (Tree Split Decomposition Fallback)"
        for k, v in feat_imp.items():
            try:
                sorted_features.append((str(k), float(v)))
            except (ValueError, TypeError):
                pass
    elif isinstance(feat_imp, list) and feat_imp:
        explanation_method_label = "Native Model Feature Importance (Tree Split Decomposition Fallback)"
        for item in feat_imp:
            if isinstance(item, dict):
                f_n = item.get("feature") or item.get("name")
                f_v = item.get("importance") or item.get("value", 0.0)
                try:
                    sorted_features.append((str(f_n), float(f_v)))
                except (ValueError, TypeError):
                    pass

    sorted_features = sorted(sorted_features, key=lambda x: abs(x[1]), reverse=True)

    if sorted_features:
        total_mag = sum(abs(v) for _, v in sorted_features) or 1.0
        exp_headers = [
            Paragraph("<b>Rank</b>", table_header),
            Paragraph("<b>Predictive Feature</b>", table_header),
            Paragraph("<b>Importance Weight</b>", table_header),
            Paragraph("<b>Attributed Dynamic / Role</b>", table_header),
        ]
        exp_rows = [exp_headers]
        for rank, (f_name, f_val) in enumerate(sorted_features[:8], start=1):
            pct_val = (abs(f_val) / total_mag) * 100.0 if total_mag > 0 else 0.0
            weight_str = f"{pct_val:.1f}% ({f_val:+.4f})" if is_shap else f"{pct_val:.1f}%"
            role_str = _attribution_role(f_name, target)
            exp_rows.append([
                Paragraph(str(rank), table_cell_center),
                Paragraph(f"<b><font color='#0284C7'>{_escape(f_name)}</font></b>", small_bold),
                Paragraph(weight_str, table_cell),
                Paragraph(_escape(role_str), table_cell),
            ])

        exp_table = Table(exp_rows, colWidths=[36, 170, 94, 240])
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EFF6FF")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(exp_table)
        story.append(Spacer(1, 2))
        story.append(Paragraph(f"<i>Methodology: {_escape(explanation_method_label)}.</i>", body_muted))
    else:
        story.append(
            Paragraph(
                f"<b>Explanations Status:</b> Not generated or unavailable for model architecture '<b>{_escape(model_name)}</b>'. "
                f"Statistical models (e.g., ARIMA) rely strictly on autoregressive lag coefficients rather than tabular feature attribution splits.",
                body_muted,
            )
        )

    # Narrative findings from parsed factors (e.g., lag_7, month, inflation_rate, ewma_3, day_of_week)
    factor_items = parsed.get("factors") or []
    if factor_items:
        story.append(Spacer(1, 3))
        for item in factor_items:
            story.append(Paragraph(f"&bull; {_format_inline_markdown(item)}", xai_bullet_style))

    story.append(Spacer(1, 4))

    # Local Explanation (LIME / Local SHAP)
    lime_data = xai.get("lime_data")
    if isinstance(lime_data, dict) and lime_data.get("features"):
        story.append(Paragraph("<b>Local Prediction Explanation (LIME Single-Observation Attribution):</b>", h3_style))
        sample_idx = lime_data.get("sample_index", 0)
        sample_ts = lime_data.get("timestamp", "Representative sample")
        story.append(Paragraph(f"<i>Representative prediction observation at timestamp {_escape(sample_ts)} (Sample index #{sample_idx}):</i>", subtitle_style))
        story.append(Spacer(1, 2))

        lime_headers = [
            Paragraph("<b>Local Feature Condition</b>", table_header),
            Paragraph("<b>Attribution Weight</b>", table_header),
            Paragraph("<b>Contribution Direction</b>", table_header),
        ]
        lime_rows = [lime_headers]
        for item in lime_data.get("features", [])[:5]:
            feat_cond = item.get("feature") or item.get("name") or "Feature"
            val = float(item.get("importance") or item.get("value") or 0.0)
            dir_str = "Positive contribution (increases predicted value)" if val >= 0 else "Negative contribution (decreases predicted value)"
            lime_rows.append([
                Paragraph(f"<b><font color='#0284C7'>{_escape(feat_cond)}</font></b>", small_bold),
                Paragraph(f"{val:+.4f}", table_cell),
                Paragraph(dir_str, table_cell),
            ])
        lime_table = Table(lime_rows, colWidths=[180, 110, 250])
        lime_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(lime_table)
        story.append(Spacer(1, 4))
    else:
        story.append(
            Paragraph(
                "<i>Local Explanation (LIME): LIME local explanation was not generated for this run. "
                "Local single-sample explanations can be computed on-demand via the /api/v1/explainability/explain endpoint.</i>",
                body_muted,
            )
        )
        story.append(Spacer(1, 4))

    # Subsection 5.5: Recommended Actions
    story.append(Paragraph("<b>Recommended Actions</b>", xai_subheading_style))
    action_items = parsed.get("actions") or []
    if action_items:
        for action in action_items:
            story.append(Paragraph(f"&bull; {_format_inline_markdown(action)}", xai_bullet_style))
    else:
        if domain.lower() == "electricity":
            story.append(Paragraph("&bull; Maintain adequate generation capacity and verify spinning reserve margins during peak demand hours.", xai_bullet_style))
            story.append(Paragraph("&bull; Continuously monitor transmission load balancing constraints during periodic diurnal transitions.", xai_bullet_style))
        else: # Retail
            story.append(Paragraph("&bull; Optimize product replenishment and safety stock buffers in advance of expected weekly peak periods.", xai_bullet_style))
            story.append(Paragraph("&bull; Review promotional discount elasticity and order lead times to prevent stockouts and holding cost overhead.", xai_bullet_style))
    story.append(Spacer(1, 4))

    # Subsection 5.6: Precautions and Monitoring
    story.append(Paragraph("<b>Precautions and Monitoring</b>", xai_subheading_style))
    precaution_items = parsed.get("precautions") or []
    if precaution_items:
        for precaution in precaution_items:
            story.append(Paragraph(f"&bull; {_format_inline_markdown(precaution)}", xai_bullet_style))
    else:
        story.append(Paragraph("&bull; Continuously track incoming actual values against predicted intervals to identify residual drift.", xai_bullet_style))
        story.append(Paragraph("&bull; Trigger formal model reassessment and retraining if cumulative forecast errors exceed operational tolerance thresholds.", xai_bullet_style))
    story.append(Spacer(1, 4))

    # Subsection 5.7: Explainability Disclosures & Limitations
    story.append(Paragraph("<b>Explainability Disclosures & Limitations</b>", xai_subheading_style))
    story.append(
        Paragraph(
            "&bull; <b>Statistical Association vs. Causal Impact:</b> Feature attributions represent statistical predictive associations within the trained model and do NOT imply physical causation or causal intervention impact.",
            xai_disclosure_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Scope Distinction:</b> Global explanations reflect aggregate model behavior across the evaluation dataset, whereas local explanations apply solely to the specific referenced prediction observation.",
            xai_disclosure_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Directional Attribution vs. Importance Magnitude:</b> Feature sign (+/-) indicates directional influence on the local prediction output, whereas global weights summarize overall predictive importance magnitude.",
            xai_disclosure_style,
        )
    )
    story.append(Spacer(1, 8))

    # Section 6: Methodology, Data Integrity & Leakage Prevention
    story.append(Paragraph("6. Academic Methodology & Data Integrity Guarantees", h2_style))
    story.append(
        Paragraph(
            "<b>Chronological Integrity:</b> Dataset splits follow strict time boundaries: Training &rarr; Validation &rarr; Held-out Test. "
            "No future observations are accessible during lag, rolling mean, or calendar feature generation.<br/>"
            "<b>Model Selection:</b> Automated model selection is performed on the validation set using lowest RMSE. "
            "Held-out test performance is reported solely for generalization assessment and is never used to select models.<br/>"
            "<b>Future Scenario Forecaster:</b> Future forecasts recursively advance lag states using previous model predictions. "
            "Zero synthetic ground truth is generated or claimed for future horizons.",
            body_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer


def generate_pdf_for_run_id(run_id: str, *, history_root: Path | str | None = None) -> io.BytesIO:
    """Load report record by run_id and return a compiled PDF BytesIO stream."""
    try:
        report = get_report(run_id)
    except Exception as err:
        raise ForecastHistoryError(f"Report '{run_id}' not found for PDF generation: {err}", code=404) from err

    return build_pdf_document(report)
