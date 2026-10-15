"""Reports routes backed by forecast history metadata."""

from flask import jsonify, request

from ...services.app_summary_service import AppSummaryError, get_report, list_reports
from . import reports_bp


def _error_response(err: AppSummaryError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


@reports_bp.get("")
def reports_list():
    try:
        limit_raw = request.args.get("limit", 20)
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError) as err:
            raise AppSummaryError("Parameter 'limit' must be an integer between 1 and 100") from err

        data = list_reports(domain=request.args.get("domain"), limit=limit)
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500


@reports_bp.get("/<run_id>")
def reports_get(run_id: str):
    try:
        data = get_report(run_id)
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500


@reports_bp.get("/<run_id>/pdf")
def reports_pdf(run_id: str):
    """Download a compiled PDF audit report for a forecast run."""
    try:
        from flask import send_file
        from ...services.pdf_report_service import generate_pdf_for_run_id
        from ml_engine.history.forecast_history import ForecastHistoryError

        pdf_buffer = generate_pdf_for_run_id(run_id)
        filename = f"SmartXSense_Report_{run_id}.pdf"
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except (AppSummaryError, ForecastHistoryError) as err:
        code = getattr(err, "code", 400)
        return jsonify({"status": "error", "code": code, "message": str(err)}), code
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": f"PDF generation error: {err}"}), 500


@reports_bp.get("/<run_id>/export/<fmt>")
def reports_export(run_id: str, fmt: str):
    """Export report forecast observations as CSV or full report as JSON."""
    try:
        from flask import Response
        import json

        data = get_report(run_id)
        fmt_clean = fmt.lower().strip()

        if fmt_clean == "json":
            json_text = json.dumps(data, indent=2, default=str)
            return Response(
                json_text,
                mimetype="application/json",
                headers={"Content-Disposition": f"attachment; filename=SmartXSense_Report_{run_id}.json"},
            )
        elif fmt_clean == "csv":
            import csv
            import io

            f_data = data.get("forecast_data") or {}
            points = f_data.get("points") or []
            f_type = data.get("forecast_type", "test_evaluation")

            buf = io.StringIO()
            writer = csv.writer(buf)
            if f_type == "test_evaluation":
                writer.writerow(["timestamp", "predicted", "actual", "residual"])
                for pt in points:
                    writer.writerow([
                        pt.get("timestamp", ""),
                        pt.get("predicted", ""),
                        pt.get("actual", ""),
                        pt.get("residual", ""),
                    ])
            else:
                writer.writerow(["timestamp", "predicted"])
                for pt in points:
                    writer.writerow([
                        pt.get("timestamp", ""),
                        pt.get("predicted", ""),
                    ])

            return Response(
                buf.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename=SmartXSense_Forecast_{run_id}.csv"},
            )
        else:
            return jsonify({"status": "error", "code": 400, "message": f"Unsupported export format '{fmt}'. Supported: 'csv', 'json'"}), 400

    except (AppSummaryError, ForecastHistoryError) as err:
        code = getattr(err, "code", 400)
        return jsonify({"status": "error", "code": code, "message": str(err)}), code
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": f"Export error: {err}"}), 500

