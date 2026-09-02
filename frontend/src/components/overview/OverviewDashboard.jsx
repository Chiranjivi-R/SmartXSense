import React, { useEffect, useState } from 'react';
import { fetchDashboardSummary } from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  secondary: '#7B61FF',
  accent: '#FF6B6B',
  success: '#00E5A0',
  warning: '#FFB800',
  bg: { dark: '#060B1A', card: '#0D1530', border: '#1E2D5A' },
  text: { primary: '#E8F0FF', secondary: '#8899CC', muted: '#4A5B8A' },
};

const formatMape = (val) => {
  if (val === undefined || val === null || Number.isNaN(Number(val))) return '—';
  const num = Number(val);
  const pct = num < 50 ? num * 100 : num;
  return `${pct.toFixed(2)}%`;
};

const formatNumber = (val, decimals = 2) => {
  if (val === undefined || val === null || Number.isNaN(Number(val))) return '—';
  return Number(val).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

const formatTime = (ts) => {
  if (!ts) return '—';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export default function OverviewDashboard({ onNavigate } = {}) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchDashboardSummary({ recentLimit: 10 });
      setData(res);
    } catch (err) {
      setError(err.message || 'Unable to load dashboard summary.');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const latest = data?.latest_forecast;
  const metrics = data?.latest_metrics || {};
  const domainDist = data?.domain_distribution || {};

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, color: COLORS.text.primary }}>Overview Dashboard</h2>
          <p style={{ margin: '6px 0 0', fontSize: 13, color: COLORS.text.secondary }}>
            Live summary from forecast history and application capabilities.
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          style={{
            background: COLORS.bg.dark,
            border: `1px solid ${COLORS.bg.border}`,
            color: COLORS.primary,
            borderRadius: 10,
            padding: '8px 14px',
            fontWeight: 700,
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: 14, color: COLORS.accent, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {loading && !data && (
        <div style={{ color: COLORS.text.secondary, padding: '24px 0' }}>Loading dashboard summary…</div>
      )}

      {data && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14, marginBottom: 20 }}>
            {[
              { label: 'Total Forecast Runs', value: data.total_forecast_runs ?? 0, color: COLORS.primary },
              { label: 'Latest Domain', value: data.latest_domain || '—', color: COLORS.secondary },
              { label: 'Latest Model', value: (data.latest_model || '—').replace(/_/g, ' '), color: COLORS.success },
              { label: 'Latest RMSE', value: formatNumber(metrics.rmse), color: COLORS.warning },
              { label: 'Latest MAPE', value: formatMape(metrics.mape), color: COLORS.accent },
              { label: 'Future Forecasts', value: data.future_forecast_runs ?? 0, color: COLORS.primary },
            ].map((card) => (
              <div key={card.label} style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 16 }}>
                <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{card.label}</div>
                <div style={{ marginTop: 8, fontSize: 22, fontWeight: 800, color: card.color, textTransform: 'capitalize' }}>{card.value}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16, marginBottom: 20 }}>
            <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18 }}>
              <div style={{ fontWeight: 700, marginBottom: 12, color: COLORS.text.primary }}>Recent Forecast Runs</div>
              {(data.recent_runs || []).length === 0 ? (
                <div style={{ color: COLORS.text.muted, fontSize: 13 }}>No forecast runs yet. Run a forecast to populate history.</div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                    <thead>
                      <tr style={{ color: COLORS.text.secondary, textAlign: 'left' }}>
                        <th style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>Time</th>
                        <th style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>Domain</th>
                        <th style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>Model</th>
                        <th style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>Type</th>
                        <th style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>RMSE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.recent_runs.map((run) => (
                        <tr key={run.run_id}>
                          <td style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>{formatTime(run.timestamp)}</td>
                          <td style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}`, textTransform: 'capitalize' }}>{run.domain}</td>
                          <td style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}`, textTransform: 'capitalize' }}>{String(run.model || '').replace(/_/g, ' ')}</td>
                          <td style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>
                            {run.forecast_type === 'future' ? 'Future' : 'Test'}
                          </td>
                          <td style={{ padding: '8px 6px', borderBottom: `1px solid ${COLORS.bg.border}` }}>
                            {run.metrics?.rmse != null ? formatNumber(run.metrics.rmse) : (run.horizon != null ? `h=${run.horizon}` : '—')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18 }}>
              <div style={{ fontWeight: 700, marginBottom: 12, color: COLORS.text.primary }}>Domain Distribution</div>
              {Object.keys(domainDist).length === 0 ? (
                <div style={{ color: COLORS.text.muted, fontSize: 13 }}>No domain activity yet.</div>
              ) : (
                Object.entries(domainDist).map(([domain, count]) => (
                  <div key={domain} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: `1px solid ${COLORS.bg.border}`, fontSize: 13 }}>
                    <span style={{ textTransform: 'capitalize', color: COLORS.text.secondary }}>{domain}</span>
                    <span style={{ fontWeight: 700, color: COLORS.primary }}>{count}</span>
                  </div>
                ))
              )}

              {latest && (
                <div style={{ marginTop: 18 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8, color: COLORS.text.primary }}>Latest Run</div>
                  <div style={{ fontSize: 12, color: COLORS.text.secondary, lineHeight: 1.6 }}>
                    <div>Type: <strong style={{ color: COLORS.text.primary }}>{latest.forecast_type}</strong></div>
                    <div>Artifact: <strong style={{ color: COLORS.text.primary }}>{latest.artifact_source || '—'}</strong></div>
                    <div>Features: <strong style={{ color: COLORS.text.primary }}>{latest.features_id || '—'}</strong></div>
                    <div>Points: <strong style={{ color: COLORS.text.primary }}>{latest.prediction_count ?? '—'}</strong></div>
                  </div>
                </div>
              )}

              {typeof onNavigate === 'function' && (
                <button
                  type="button"
                  onClick={() => onNavigate('forecast')}
                  style={{
                    marginTop: 16,
                    width: '100%',
                    border: 'none',
                    borderRadius: 10,
                    padding: '10px 14px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    color: '#fff',
                    background: `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.secondary})`,
                  }}
                >
                  Open Forecasts →
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
