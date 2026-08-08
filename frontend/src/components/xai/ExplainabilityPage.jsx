import React, { useState } from 'react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { explainForecast, DOMAIN_FEATURE_ARTIFACTS } from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  secondary: '#7B61FF',
  accent: '#FF6B6B',
  success: '#00E5A0',
  bg: { dark: '#060B1A', card: '#0D1530', border: '#1E2D5A' },
  text: { primary: '#E8F0FF', secondary: '#8899CC', muted: '#4A5B8A' },
};

const tooltipStyle = {
  backgroundColor: '#0D1530',
  border: '1px solid #1E2D5A',
  borderRadius: 10,
  color: '#E8F0FF',
  fontSize: 12,
};

const formatFeature = (name) => String(name || '')
  .replace(/_/g, ' ')
  .replace(/\b\w/g, (c) => c.toUpperCase());

export default function ExplainabilityPage() {
  const [domain, setDomain] = useState('retail');
  const [model, setModel] = useState('random_forest');
  const [method, setMethod] = useState('shap');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const runExplain = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await explainForecast({
        domain,
        model,
        featuresDir: DOMAIN_FEATURE_ARTIFACTS[domain],
        explanationType: method === 'lime' ? 'local' : 'global',
        method,
      });
      setData(res);
    } catch (err) {
      setError(err.message || 'Explainability request failed.');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const importances = (() => {
    if (!data) return [];
    if (data.feature_importance) {
      return Object.entries(data.feature_importance).map(([feature, importance]) => ({
        feature: formatFeature(feature),
        importance: Math.abs(Number(importance)),
      }));
    }
    if (Array.isArray(data.feature_importances)) {
      return data.feature_importances.map((item) => ({
        feature: formatFeature(item.feature),
        importance: Math.abs(Number(item.importance)),
      }));
    }
    if (Array.isArray(data.feature_attributions)) {
      return data.feature_attributions.map((item) => ({
        feature: formatFeature(item.rule || item.feature),
        importance: Math.abs(Number(item.weight ?? item.importance)),
      }));
    }
    return [];
  })();

  const top = importances.slice(0, 12);

  return (
    <div>
      <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, color: COLORS.text.primary }}>Explainable AI</h2>
      <p style={{ margin: '6px 0 20px', fontSize: 13, color: COLORS.text.secondary }}>
        Live SHAP/LIME explanations from the existing explainability API (Random Forest & XGBoost).
      </p>

      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18, marginBottom: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
          <label style={{ fontSize: 12, color: COLORS.text.secondary }}>
            Domain
            <select value={domain} onChange={(e) => setDomain(e.target.value)} style={selectStyle}>
              <option value="retail">Retail</option>
              <option value="electricity">Electricity</option>
            </select>
          </label>
          <label style={{ fontSize: 12, color: COLORS.text.secondary }}>
            Model
            <select value={model} onChange={(e) => setModel(e.target.value)} style={selectStyle}>
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
              <option value="lstm">LSTM (unsupported)</option>
            </select>
          </label>
          <label style={{ fontSize: 12, color: COLORS.text.secondary }}>
            Method
            <select value={method} onChange={(e) => setMethod(e.target.value)} style={selectStyle}>
              <option value="shap">SHAP (global)</option>
              <option value="lime">LIME (local sample 0)</option>
            </select>
          </label>
          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button type="button" onClick={runExplain} disabled={loading} style={buttonStyle(loading)}>
              {loading ? 'Computing…' : 'Run Explanation'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: 14, color: COLORS.accent, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {!data && !loading && !error && (
        <div style={{ color: COLORS.text.muted, fontSize: 13, padding: '12px 0' }}>
          Choose domain/model and run an explanation. Results come from the live backend — no hardcoded SHAP values.
        </div>
      )}

      {data && (
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18 }}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16, fontSize: 13, color: COLORS.text.secondary }}>
            <span>Model: <strong style={{ color: COLORS.primary, textTransform: 'capitalize' }}>{String(data.model || model).replace(/_/g, ' ')}</strong></span>
            <span>Method: <strong style={{ color: COLORS.text.primary }}>{data.method || method}</strong></span>
            <span>Type: <strong style={{ color: COLORS.text.primary }}>{data.explanation_type || (method === 'lime' ? 'local' : 'global')}</strong></span>
            <span>Samples: <strong style={{ color: COLORS.text.primary }}>{data.sample_count ?? (data.sample_index != null ? 1 : '—')}</strong></span>
            <span>Domain: <strong style={{ color: COLORS.text.primary, textTransform: 'capitalize' }}>{data.domain || domain}</strong></span>
          </div>

          {top.length === 0 ? (
            <div style={{ color: COLORS.text.muted }}>No feature importances returned for this explanation.</div>
          ) : (
            <ResponsiveContainer width="100%" height={Math.max(280, top.length * 28)}>
              <BarChart layout="vertical" data={top} margin={{ top: 8, right: 20, left: 120, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                <XAxis type="number" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" width={110} tick={{ fill: COLORS.text.primary, fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="importance" name="Mean |SHAP|" radius={[0, 4, 4, 0]}>
                  {top.map((_, i) => (
                    <Cell key={i} fill={i < 3 ? COLORS.primary : COLORS.secondary} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
    </div>
  );
}

const selectStyle = {
  display: 'block',
  width: '100%',
  marginTop: 6,
  background: '#060B1A',
  border: '1px solid #1E2D5A',
  borderRadius: 10,
  padding: '10px 12px',
  color: '#E8F0FF',
  fontWeight: 600,
};

const buttonStyle = (disabled) => ({
  width: '100%',
  border: 'none',
  borderRadius: 10,
  padding: '12px 14px',
  fontWeight: 700,
  cursor: disabled ? 'not-allowed' : 'pointer',
  color: '#fff',
  background: disabled ? '#1E2D5A' : 'linear-gradient(135deg, #00D4FF, #7B61FF)',
});
