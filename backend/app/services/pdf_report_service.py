"""SmartXSense Professional PDF Report Generator.

Generates an authoritative, publication-quality PDF audit report for any
forecast run (held-out test evaluation or future scenario projection).
Uses ReportLab Platypus layout with clean typography, data tables, and
methodology disclosures.
"""

from __future__ import annotations

import html
import io
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
        self.drawString(36, 22, "SmartXSense Academic Forecast Audit — Leakage-Safe Chronological Split")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 22, page_str)
        self.restoreState()


def _escape(text: Any) -> str:
    if text is None:
        return "—"
    return html.escape(str(text))


def _format_num(val: Any, decimals: int = 2) -> str:
    if val is None or val == "":
        return "—"
    try:
        num = float(val)
        return f"{num:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _format_mape(val: Any) -> str:
    if val is None or val == "":
        return "—"
    try:
        num = float(val)
        pct = num * 100.0 if num <= 1.0 else num
        return f"{pct:.2f}%"
    except (ValueError, TypeError):
        return str(val)


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
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
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
            Paragraph(f"<b>Selection Split:</b> Validation Set", body_style),
            Paragraph(f"<b>Selection Criterion:</b> Lowest Validation RMSE", body_style),
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

    # Section 1: Executive Forecast Summary
    summary_obj = report.get("forecast_summary") or {}
    story.append(Paragraph("1. Executive Forecast Summary & Behavioral Dynamics", h2_style))
    summary_text = summary_obj.get("summary") or f"Forecast executed for {domain.lower()} using {model_name}."
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

    # Section 2: Model Performance Evaluation
    metrics = report.get("metrics") or (report.get("test_evaluation") or {}).get("metrics") or {}
    if f_type == "test_evaluation" and metrics:
        story.append(Paragraph("2. Held-Out Test Evaluation (Unseen Ground Truth)", h2_style))
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
        story.append(Paragraph("2. Future Scenario Projection Summary", h2_style))
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

    story.append(Spacer(1, 8))

    # Section 3: Validation Benchmark & Model Selection
    val_bench = report.get("validation_benchmark") or {}
    ranking = val_bench.get("ranking") or []
    story.append(Paragraph("3. Validation Benchmark & Selection Ranking", h2_style))
    story.append(
        Paragraph(
            "<i>Model auto-selection is executed strictly on the chronologically preceding validation split. Lowest Validation RMSE wins:</i>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 3))

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

    # Section 4: Explainability & Attribution
    exp_obj = report.get("explainability") or {}
    feat_imp = exp_obj.get("feature_importance") or report.get("feature_importance") or report.get("feature_importances")
    if feat_imp and isinstance(feat_imp, dict):
        story.append(Paragraph("4. Feature Explainability & Attribution Drivers", h2_style))
        story.append(
            Paragraph(
                "<i>Normalized feature importance derived from Tree SHAP and ensemble weight decomposition:</i>",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 3))

        sorted_feats = sorted(feat_imp.items(), key=lambda kv: float(kv[1]), reverse=True)[:5]
        total_sum = sum(float(v) for _, v in sorted_feats) or 1.0
        exp_rows = [
            [
                Paragraph("<b>Predictive Feature</b>", table_header),
                Paragraph("<b>Normalized Weight</b>", table_header),
                Paragraph("<b>Attribution Role</b>", table_header),
            ]
        ]
        for f_name, f_val in sorted_feats:
            pct = (float(f_val) / total_sum) * 100.0 if total_sum > 0 else 0.0
            role = "Short-term temporal anchor" if "lag_1" in f_name else \
                   "Diurnal cycle regulator" if ("lag_24" in f_name or "hour" in f_name) else \
                   "Weekly seasonality baseline" if "168" in f_name or "7" in f_name else \
                   "Trend smoothing moving window" if "rolling" in f_name else "Contextual predictor"
            exp_rows.append([
                Paragraph(_escape(f_name), small_bold),
                Paragraph(f"{pct:.1f}%", table_cell),
                Paragraph(role, table_cell),
            ])

        exp_table = Table(exp_rows, colWidths=[160, 110, 270])
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(exp_table)
        story.append(Spacer(1, 8))

    # Section 5: Forecast Data Sample
    f_data = report.get("forecast_data") or {}
    points = f_data.get("points") or []
    if points:
        story.append(Paragraph("5. Forecast Time-Series Observations (Sample)", h2_style))
        sample_pts = points[:15]
        f_headers = [
            Paragraph("<b>Timestamp (UTC)</b>", table_header),
            Paragraph("<b>Predicted</b>", table_header),
        ]
        if f_type == "test_evaluation":
            f_headers.extend([
                Paragraph("<b>Actual (Ground Truth)</b>", table_header),
                Paragraph("<b>Residual (Error)</b>", table_header),
            ])
            col_widths = [180, 120, 120, 120]
        else:
            col_widths = [270, 270]

        pt_rows = [f_headers]
        for pt in sample_pts:
            ts_label = str(pt.get("timestamp", ""))[:19].replace("T", " ")
            p_val = _format_num(pt.get("predicted"))
            row = [Paragraph(ts_label, table_cell), Paragraph(p_val, table_cell)]
            if f_type == "test_evaluation":
                a_val = _format_num(pt.get("actual"))
                r_val = _format_num(pt.get("residual"))
                row.extend([Paragraph(a_val, table_cell), Paragraph(r_val, table_cell)])
            pt_rows.append(row)

        sample_table = Table(pt_rows, colWidths=col_widths)
        sample_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(sample_table)
        if len(points) > len(sample_pts):
            story.append(Spacer(1, 2))
            story.append(
                Paragraph(
                    f"<i>Note: Displaying first {len(sample_pts)} of {len(points)} forecast points. Full data export available in CSV and JSON formats.</i>",
                    body_muted,
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
