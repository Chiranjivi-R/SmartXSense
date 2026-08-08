import React, { useEffect, useState } from 'react';
import { fetchModelsCatalog } from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  secondary: '#7B61FF',
  accent: '#FF6B6B',
  success: '#00E5A0',
  warning: '#FFB800',
  bg: { dark: '#060B1A', card: '#0D1530', border: '#1E2D5A' },
  text: { primary: '#E8F0FF', secondary: '#8899CC', muted: '#4A5B8A' },
};

const Flag = ({ ok, labelYes = 'Yes', labelNo = 'No' }) => (
  <span style={{ color: ok ? COLORS.success : COLORS.text.muted, fontWeight: 700 }}>
    {ok ? labelYes : labelNo}
  </span>
);

export default function ModelsCatalogPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await fetchModelsCatalog());
    } catch (err) {
      setError(err.message || 'Unable to load model catalog.');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, color: COLORS.text.primary }}>AI Models</h2>
          <p style={{ margin: '6px 0 0', fontSize: 13, color: COLORS.text.secondary }}>
            Capability catalog from the live backend. Accuracy metrics are not fabricated — run a forecast to evaluate.
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

      {loading && !data && <div style={{ color: COLORS.text.secondary }}>Loading model catalog…</div>}

      {data && (
        <>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16, fontSize: 12, color: COLORS.text.secondary }}>
            <span>Domains: <strong style={{ color: COLORS.text.primary }}>{(data.domains || []).join(', ')}</strong></span>
            <span>Explainable: <strong style={{ color: COLORS.text.primary }}>{(data.explainable_models || []).join(', ')}</strong></span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
            {(data.models || []).map((model) => (
              <div key={model.key} style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 16 }}>
                <div style={{ fontSize: 16, fontWeight: 800, color: COLORS.text.primary, marginBottom: 4 }}>{model.name}</div>
                <div style={{ fontSize: 12, color: COLORS.text.muted, marginBottom: 12, textTransform: 'capitalize' }}>{model.model_type?.replace(/_/g, ' ')}</div>
                {[
                  { label: 'Domains', value: (model.domains || []).join(', ') || '—' },
                  { label: 'Persistence', value: <Flag ok={model.persistence_supported} /> },
                  { label: 'Explainability', value: <Flag ok={model.explainability_supported} /> },
                  {
                    label: 'Future Forecast',
                    value: model.future_forecast_supported
                      ? `Yes (${(model.future_forecast_domains || []).join(', ')})`
                      : 'No',
                  },
                  { label: 'Evaluation', value: model.evaluation_note || 'Run forecast to evaluate.' },
                ].map((row) => (
                  <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, padding: '6px 0', borderBottom: `1px solid ${COLORS.bg.border}`, fontSize: 12 }}>
                    <span style={{ color: COLORS.text.muted }}>{row.label}</span>
                    <span style={{ color: COLORS.text.primary, textAlign: 'right', textTransform: 'capitalize' }}>{row.value}</span>
                  </div>
                ))}
                {model.notes && (
                  <div style={{ marginTop: 10, fontSize: 11, color: COLORS.warning, lineHeight: 1.4 }}>{model.notes}</div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
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
