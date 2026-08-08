import React, { useEffect, useState } from 'react';
import { fetchReport, fetchReports } from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  accent: '#FF6B6B',
  success: '#00E5A0',
  bg: { dark: '#060B1A', card: '#0D1530', border: '#1E2D5A' },
  text: { primary: '#E8F0FF', secondary: '#8899CC', muted: '#4A5B8A' },
};

const formatMape = (val) => {
  if (val == null || Number.isNaN(Number(val))) return '—';
  const num = Number(val);
  return `${(num > 1 ? num : num * 100).toFixed(2)}%`;
};

const formatNumber = (val) => {
  if (val == null || Number.isNaN(Number(val))) return '—';
  return Number(val).toFixed(2);
};

export default function ReportsPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reports, setReports] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchReports({ limit: 30 });
      setReports(res.reports || []);
    } catch (err) {
      setError(err.message || 'Unable to load reports.');
      setReports([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openReport = async (runId) => {
    setSelectedId(runId);
    setDetailLoading(true);
    setDetailError(null);
    try {
      setDetail(await fetchReport(runId));
    } catch (err) {
      setDetail(null);
      setDetailError(err.message || 'Unable to load report detail.');
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, color: COLORS.text.primary }}>Reports</h2>
          <p style={{ margin: '6px 0 0', fontSize: 13, color: COLORS.text.secondary }}>
            Forecast run reports from history metadata (no prediction arrays stored).
          </p>
        </div>
        <button type="button" onClick={load} disabled={loading} style={refreshStyle(loading)}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: 14, color: COLORS.accent, marginBottom: 16 }}>
          {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 16 }}>
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 16 }}>
          {loading && reports.length === 0 && <div style={{ color: COLORS.text.secondary }}>Loading reports…</div>}
          {!loading && reports.length === 0 && !error && (
            <div style={{ color: COLORS.text.muted, fontSize: 13 }}>No reports yet. Successful forecasts appear here automatically.</div>
          )}
          {reports.map((report) => (
            <button
              key={report.run_id}
              type="button"
              onClick={() => openReport(report.run_id)}
              style={{
                width: '100%',
                textAlign: 'left',
                background: selectedId === report.run_id ? `${COLORS.primary}14` : 'transparent',
                border: `1px solid ${selectedId === report.run_id ? COLORS.primary : COLORS.bg.border}`,
                borderRadius: 10,
                padding: 12,
                marginBottom: 8,
                cursor: 'pointer',
                color: COLORS.text.primary,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'capitalize' }}>
                {report.domain} · {String(report.model || '').replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: 11, color: COLORS.text.secondary, marginTop: 4 }}>
                {report.forecast_type} · {report.timestamp ? new Date(report.timestamp).toLocaleString() : '—'}
              </div>
              <div style={{ fontSize: 11, color: COLORS.text.muted, marginTop: 4 }}>
                RMSE {formatNumber(report.metrics?.rmse)} · MAPE {formatMape(report.metrics?.mape)}
              </div>
            </button>
          ))}
        </div>

        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 16 }}>
          <div style={{ fontWeight: 700, marginBottom: 12 }}>Report Detail</div>
          {!selectedId && <div style={{ color: COLORS.text.muted, fontSize: 13 }}>Select a report to view metadata.</div>}
          {detailLoading && <div style={{ color: COLORS.text.secondary }}>Loading detail…</div>}
          {detailError && <div style={{ color: COLORS.accent, fontSize: 13 }}>{detailError}</div>}
          {detail && !detailLoading && (
            <div style={{ fontSize: 13, color: COLORS.text.secondary, lineHeight: 1.7 }}>
              {[
                ['Run ID', detail.run_id],
                ['Domain', detail.domain],
                ['Model', detail.model],
                ['Target', detail.target],
                ['Type', detail.forecast_type],
                ['Artifact', detail.artifact_source || '—'],
                ['Features ID', detail.features_id || '—'],
                ['Predictions', detail.prediction_count],
                ['Horizon', detail.horizon ?? '—'],
                ['MAE', formatNumber(detail.metrics?.mae)],
                ['RMSE', formatNumber(detail.metrics?.rmse)],
                ['MAPE', formatMape(detail.metrics?.mape)],
                ['R²', formatNumber(detail.metrics?.r2)],
              ].map(([label, value]) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, borderBottom: `1px solid ${COLORS.bg.border}`, padding: '6px 0' }}>
                  <span style={{ color: COLORS.text.muted }}>{label}</span>
                  <span style={{ color: COLORS.text.primary, textAlign: 'right', wordBreak: 'break-all' }}>{String(value)}</span>
                </div>
              ))}
              {'predictions' in detail && (
                <div style={{ marginTop: 10, color: COLORS.accent }}>Unexpected predictions payload detected — should not be stored in reports.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const refreshStyle = (disabled) => ({
  background: '#060B1A',
  border: '1px solid #1E2D5A',
  color: '#00D4FF',
  borderRadius: 10,
  padding: '8px 14px',
  fontWeight: 700,
  cursor: disabled ? 'not-allowed' : 'pointer',
});
