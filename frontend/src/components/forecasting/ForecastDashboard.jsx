import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { runForecast, compareModels, explainForecast } from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  secondary: '#7B61FF',
  accent: '#FF6B6B',
  success: '#00E5A0',
  warning: '#FFB800',
  bg: {
    dark: '#060B1A',
    card: '#0D1530',
    cardHover: '#121B3A',
    border: '#1E2D5A',
  },
  text: {
    primary: '#E8F0FF',
    secondary: '#8899CC',
    muted: '#4A5B8A',
  },
};

const customTooltipStyle = {
  backgroundColor: '#0D1530',
  border: '1px solid #1E2D5A',
  borderRadius: 10,
  color: '#E8F0FF',
  fontSize: 13,
  padding: '12px 16px',
  boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
};

const formatTimestamp = (ts) => {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch (e) {
    return String(ts);
  }
};

const formatNumber = (val, decimals = 2) => {
  if (val === undefined || val === null || isNaN(val)) return '—';
  return Number(val).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

const formatMape = (val) => {
  if (val === undefined || val === null || isNaN(val)) return '—';
  const pct = val > 1 ? val : val * 100;
  return `${pct.toFixed(2)}%`;
};

const formatFeatureName = (name) => {
  if (!name) return '';
  const dictionary = {
    city_tier: 'City Tier',
    cpi_clothing: 'CPI Clothing',
    cpi_combined: 'CPI Combined',
    city: 'City',
    category: 'Product Category',
    temperature_c: 'Temperature (°C)',
    humidity: 'Humidity (%)',
    demand_mw: 'Demand (MW)',
    sales: 'Sales Units',
    rolling_mean_7: 'Rolling Mean (7d)',
    rolling_mean_30: 'Rolling Mean (30d)',
    lag_1: 'Lag 1',
    lag_7: 'Lag 7',
    day_of_week: 'Day of Week',
    is_weekend: 'Weekend Flag',
    holiday_flag: 'Holiday Flag',
  };
  if (dictionary[name]) return dictionary[name];

  return String(name)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const DOMAIN_CONTEXTS = {
  retail: 'Forecasting retail sales demand using historical sales, pricing, economic indicators, calendar signals, and lag features.',
  electricity: 'Forecasting electricity grid demand using temporal patterns, historical load, and rolling aggregate features.',
};

export default function ForecastDashboard({ initialDomain = 'retail' }) {
  const [domain, setDomain] = useState(initialDomain);
  const [model, setModel] = useState('random_forest');

  // State Management
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [forecastError, setForecastError] = useState(null);

  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData, setCompareData] = useState(null);
  const [compareError, setCompareError] = useState(null);

  const [autoSelectedModel, setAutoSelectedModel] = useState(null);

  const [explainLoading, setExplainLoading] = useState(false);
  const [explainData, setExplainData] = useState(null);
  const [explainError, setExplainError] = useState(null);

  // Sync state if initialDomain prop changes
  useEffect(() => {
    handleDomainSwitch(initialDomain);
  }, [initialDomain]);

  // Initial forecast on mount
  useEffect(() => {
    handleRunForecast('random_forest', initialDomain);
  }, []);

  // Clear stale state when domain changes
  const handleDomainSwitch = (newDomain) => {
    setDomain(newDomain);
    setForecastData(null);
    setCompareData(null);
    setExplainData(null);
    setAutoSelectedModel(null);
    setForecastError(null);
    setCompareError(null);
    setExplainError(null);
    handleRunForecast(model, newDomain);
  };

  // Primary forecast runner
  const handleRunForecast = async (targetModel = model, targetDomain = domain) => {
    setForecastLoading(true);
    setForecastError(null);
    setExplainData(null);
    setExplainError(null);

    let actualModel = targetModel;
    let recModel = null;

    try {
      // Auto Select workflow
      if (targetModel === 'auto_select') {
        setCompareLoading(true);
        try {
          const compResult = await compareModels({ domain: targetDomain });
          setCompareData(compResult);
          actualModel = compResult.recommended_model || 'random_forest';
          recModel = actualModel;
          setAutoSelectedModel(recModel);
        } catch (err) {
          console.error('Auto-select comparison failed:', err);
          actualModel = 'random_forest';
        } finally {
          setCompareLoading(false);
        }
      } else {
        setAutoSelectedModel(null);
      }

      // Execute forecast call
      const res = await runForecast({
        domain: targetDomain,
        model: actualModel,
        predictionLimit: 100,
      });

      setForecastData(res);

      // Trigger SHAP explainability for supported tree models
      if (['random_forest', 'xgboost'].includes(actualModel)) {
        fetchExplainability(targetDomain, actualModel);
      }
    } catch (err) {
      console.error('Forecast error:', err);
      setForecastError('Forecast request failed. Please verify that the SmartXSense backend server is running.');
    } finally {
      setForecastLoading(false);
    }
  };

  // Model comparison runner
  const handleCompareModels = async () => {
    setCompareLoading(true);
    setCompareError(null);
    try {
      const res = await compareModels({ domain });
      setCompareData(res);
    } catch (err) {
      console.error('Comparison error:', err);
      setCompareError('Model comparison request failed. Please verify backend service.');
    } finally {
      setCompareLoading(false);
    }
  };

  // Explainability runner
  const fetchExplainability = async (expDomain, expModel) => {
    setExplainLoading(true);
    setExplainError(null);
    try {
      const res = await explainForecast({
        domain: expDomain,
        model: expModel,
        explanationType: 'global',
        method: 'shap',
      });
      setExplainData(res);
    } catch (err) {
      console.error('Explainability error:', err);
      setExplainError('Explainability request failed for this model.');
    } finally {
      setExplainLoading(false);
    }
  };

  // Prepare chart series from predictions[]
  const chartData = forecastData?.predictions?.map((item) => {
    const act = item.actual !== undefined && item.actual !== null ? Number(item.actual) : null;
    const pred = item.predicted !== undefined && item.predicted !== null ? Number(item.predicted) : null;
    const absErr = act !== null && pred !== null ? Math.abs(act - pred) : null;
    return {
      rawTimestamp: item.timestamp,
      timestamp: formatTimestamp(item.timestamp),
      actual: act,
      predicted: pred,
      absError: absErr,
    };
  }) || [];

  const activeModelDisplay = autoSelectedModel
    ? `${autoSelectedModel.replace('_', ' ')} (Auto Selected)`
    : (forecastData?.model || model).replace('_', ' ');

  const isExplainableModel = ['random_forest', 'xgboost'].includes(forecastData?.model || model);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* ─── 1. Control Header & Domain Context ─── */}
      <div
        style={{
          background: COLORS.bg.card,
          border: `1px solid ${COLORS.bg.border}`,
          borderRadius: 16,
          padding: 24,
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 24, fontWeight: 800, margin: 0, color: COLORS.text.primary, fontFamily: "'Space Grotesk', sans-serif" }}>
              SmartXSense Forecast Intelligence
            </h1>
            <p style={{ fontSize: 13, color: COLORS.text.secondary, margin: '4px 0 0', maxWidth: 800, lineHeight: 1.5 }}>
              {DOMAIN_CONTEXTS[domain]}
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => handleRunForecast(model, domain)}
              disabled={forecastLoading || compareLoading}
              style={{
                background: forecastLoading
                  ? COLORS.bg.border
                  : `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.secondary})`,
                color: '#fff',
                border: 'none',
                borderRadius: 10,
                padding: '10px 22px',
                fontSize: 14,
                fontWeight: 700,
                cursor: forecastLoading ? 'not-allowed' : 'pointer',
                boxShadow: forecastLoading ? 'none' : `0 4px 20px ${COLORS.primary}44`,
                transition: 'all 0.2s',
              }}
            >
              {forecastLoading ? 'Generating Forecast...' : '⚡ Run Forecast'}
            </button>
            <button
              onClick={handleCompareModels}
              disabled={compareLoading}
              style={{
                background: 'transparent',
                color: COLORS.primary,
                border: `1px solid ${COLORS.primary}`,
                borderRadius: 10,
                padding: '10px 20px',
                fontSize: 14,
                fontWeight: 700,
                cursor: compareLoading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {compareLoading ? 'Benchmarking...' : '📊 Compare Models'}
            </button>
          </div>
        </div>

        {/* Form Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
          {/* Domain Selector */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Target Domain
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              {[
                { id: 'retail', label: '🛒 Retail Demand', sub: 'Sales & Inventory' },
                { id: 'electricity', label: '⚡ Electricity Load', sub: 'Grid System MW' },
              ].map((d) => (
                <button
                  key={d.id}
                  onClick={() => handleDomainSwitch(d.id)}
                  style={{
                    flex: 1,
                    padding: '10px 14px',
                    borderRadius: 10,
                    border: `1px solid ${domain === d.id ? COLORS.primary : COLORS.bg.border}`,
                    background: domain === d.id ? `${COLORS.primary}18` : COLORS.bg.dark,
                    color: domain === d.id ? COLORS.primary : COLORS.text.secondary,
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontWeight: 700,
                    fontSize: 13,
                    transition: 'all 0.2s',
                  }}
                >
                  <div>{d.label}</div>
                  <div style={{ fontSize: 11, color: COLORS.text.muted, fontWeight: 400, marginTop: 2 }}>{d.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Model Selector */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Forecasting Model
            </label>
            <select
              value={model}
              onChange={(e) => {
                const selected = e.target.value;
                setModel(selected);
                handleRunForecast(selected, domain);
              }}
              style={{
                width: '100%',
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                borderRadius: 10,
                padding: '12px 16px',
                color: COLORS.text.primary,
                fontSize: 14,
                fontWeight: 600,
                outline: 'none',
              }}
            >
              <option value="random_forest">Random Forest (Tree Ensemble)</option>
              <option value="xgboost">XGBoost (Gradient Boosted Trees)</option>
              <option value="hybrid">Hybrid Ensemble (Validation-Weighted)</option>
              <option value="auto_select">⚡ Auto Select (Validation Recommended)</option>
              {domain === 'electricity' && <option value="arima">ARIMA (Time Series)</option>}
            </select>
          </div>

          {/* Active Status Display */}
          <div style={{ background: COLORS.bg.dark, borderRadius: 10, padding: '12px 16px', border: `1px solid ${COLORS.bg.border}`, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Demo Dataset Status</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginTop: 4, fontFamily: "'Space Mono', monospace" }}>
              Demo Dataset: {domain.toUpperCase()}
            </div>
            <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2 }}>
              Active Model: <span style={{ color: COLORS.primary, fontWeight: 700, textTransform: 'capitalize' }}>{activeModelDisplay}</span>
            </div>
          </div>
        </div>

        {/* ─── Auto Select Explanation Banner ─── */}
        {autoSelectedModel && (
          <div style={{ background: `${COLORS.success}15`, border: `1px solid ${COLORS.success}44`, borderRadius: 10, padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 14 }}>
            <span style={{ fontSize: 24 }}>🏆</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.success }}>
                Auto Select Recommendation: <span style={{ textTransform: 'capitalize' }}>{autoSelectedModel.replace('_', ' ')}</span>
              </div>
              <div style={{ fontSize: 13, color: COLORS.text.primary, marginTop: 2, lineHeight: 1.4 }}>
                SmartXSense compared candidate models using validation RMSE and selected <strong>{autoSelectedModel.replace('_', ' ').toUpperCase()}</strong> as the best-performing candidate. The held-out test set remains reserved strictly for final evaluation.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Presentation-Safe Error Alert */}
      {forecastError && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: '16px 20px', color: COLORS.accent, fontSize: 14, fontWeight: 600 }}>
          ⚠️ {forecastError}
        </div>
      )}

      {/* ─── 2. Compact Forecast Summary Header ─── */}
      {forecastData && (
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <span style={{ fontSize: 16, fontWeight: 800, color: COLORS.text.primary }}>
              {domain === 'retail' ? 'Retail Sales Demand Forecast' : 'Electricity System Load Forecast'}
            </span>
            <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2 }}>
              Target Variable: <span style={{ color: COLORS.primary, fontWeight: 700 }}>{forecastData.target}</span> • Dataset: <span style={{ color: COLORS.text.primary }}>Demo Dataset ({domain})</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, color: COLORS.primary, padding: '4px 12px', borderRadius: 8, fontSize: 12, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>
              Model: {forecastData.model?.replace('_', ' ').toUpperCase()}
            </span>
            <span style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, color: COLORS.success, padding: '4px 12px', borderRadius: 8, fontSize: 12, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>
              {forecastData.prediction_count || chartData.length} Forecast Points
            </span>
          </div>
        </div>
      )}

      {/* ─── 3. Metric Cards with Tooltip Explanations ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
        {[
          {
            label: 'MAE',
            title: 'Mean Absolute Error',
            value: formatNumber(forecastData?.test_metrics?.mae),
            desc: 'Average absolute prediction error between forecast and actuals.',
            color: COLORS.primary,
          },
          {
            label: 'RMSE',
            title: 'Root Mean Squared Error',
            value: formatNumber(forecastData?.test_metrics?.rmse),
            desc: 'Root mean squared error penalizing larger forecasting deviations.',
            color: COLORS.secondary,
          },
          {
            label: 'MAPE',
            title: 'Mean Abs % Error',
            value: formatMape(forecastData?.test_metrics?.mape),
            desc: 'Average percentage forecasting error across test horizon.',
            color: COLORS.success,
          },
          {
            label: 'R² Score',
            title: 'Coefficient of Determination',
            value: formatNumber(forecastData?.test_metrics?.r2, 4),
            desc: 'Explained variance / goodness of fit (1.0 = perfect fit).',
            color: COLORS.warning,
          },
        ].map((m, idx) => (
          <div
            key={idx}
            style={{
              background: COLORS.bg.card,
              border: `1px solid ${COLORS.bg.border}`,
              borderRadius: 16,
              padding: '20px',
              position: 'relative',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              justify: 'space-between',
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.text.secondary, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  {m.label}
                </span>
                <span style={{ fontSize: 11, color: COLORS.text.muted }} title={m.desc}>ℹ️</span>
              </div>
              <div style={{ fontSize: 26, fontWeight: 800, color: m.color, fontFamily: "'Space Mono', monospace", margin: '6px 0' }}>
                {forecastLoading ? '...' : m.value}
              </div>
            </div>
            <div style={{ fontSize: 11, color: COLORS.text.muted, lineHeight: 1.4, marginTop: 4 }}>
              {m.desc}
            </div>
          </div>
        ))}
      </div>

      {/* ─── Performance Interpretation Banner ─── */}
      {forecastData?.test_metrics && (
        <div style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 12, padding: '12px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <div style={{ fontSize: 13, color: COLORS.text.secondary }}>
            📊 <strong style={{ color: COLORS.text.primary }}>Performance Interpretation (Held-Out Test Set):</strong> Test MAPE is <span style={{ color: COLORS.success, fontWeight: 700 }}>{formatMape(forecastData.test_metrics.mape)}</span> with R² of <span style={{ color: COLORS.warning, fontWeight: 700 }}>{formatNumber(forecastData.test_metrics.r2, 4)}</span>.
          </div>
          <span style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Evaluation Split: Held-Out Test Data
          </span>
        </div>
      )}

      {/* ─── 4. Hybrid Component Weights Visualization ─── */}
      {forecastData?.model === 'hybrid' && forecastData?.weights && (
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 4 }}>
            Ensemble Component Weights (Validation Derived)
          </div>
          <div style={{ fontSize: 12, color: COLORS.text.secondary, marginBottom: 14 }}>
            Hybrid ensemble weights are derived automatically from inverse validation RMSE performance.
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            {Object.entries(forecastData.weights).map(([comp, w]) => {
              const pct = Number(w) * 100;
              return (
                <div key={comp} style={{ background: COLORS.bg.dark, borderRadius: 10, padding: '10px 14px', border: `1px solid ${COLORS.bg.border}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 700, color: COLORS.text.primary, marginBottom: 6 }}>
                    <span style={{ textTransform: 'capitalize' }}>{comp.replace('_', ' ')}</span>
                    <span style={{ color: COLORS.primary, fontFamily: "'Space Mono', monospace" }}>{pct.toFixed(1)}%</span>
                  </div>
                  <div style={{ background: COLORS.bg.border, height: 6, borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: `linear-gradient(90deg, ${COLORS.primary}, ${COLORS.secondary})`, borderRadius: 3 }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ─── 5. Actual vs Predicted Line Chart ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Actual vs Predicted Demand Series (Held-Out Test Set)
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '2px 0 0' }}>
              Comparison of actual ground truth observations against model forecasts
            </p>
          </div>
          {forecastData?.total_test_predictions && (
            <span style={{ fontSize: 12, color: COLORS.text.muted, fontFamily: "'Space Mono', monospace" }}>
              Showing {chartData.length} of {forecastData.total_test_predictions} test observations
            </span>
          )}
        </div>

        {forecastLoading ? (
          <div style={{ height: 320, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: COLORS.text.secondary }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>⚡</div>
            <div>Executing model inference & computing predictions...</div>
          </div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} opacity={0.6} />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: COLORS.text.muted, fontSize: 11 }}
                interval="preserveStartEnd"
                minTickGap={20}
              />
              <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={customTooltipStyle}
                formatter={(val, name, entry) => {
                  if (name === 'Actual Ground Truth' || name === 'Predicted Forecast') {
                    return [formatNumber(val), name];
                  }
                  return [formatNumber(val), name];
                }}
                labelFormatter={(label, payload) => {
                  const pt = payload && payload[0]?.payload;
                  if (pt && pt.absError !== null) {
                    return `${label} (Abs Error: ${formatNumber(pt.absError)})`;
                  }
                  return label;
                }}
              />
              <Legend wrapperStyle={{ fontSize: 13, color: COLORS.text.secondary, paddingTop: 10 }} />
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual Ground Truth"
                stroke={COLORS.primary}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Predicted Forecast"
                stroke={COLORS.accent}
                strokeWidth={2.5}
                strokeDasharray="4 4"
                dot={false}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ height: 260, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: COLORS.text.muted }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>📊</div>
            <div>Select a domain and forecasting model, then run a forecast to view model performance and predictions.</div>
          </div>
        )}
      </div>

      {/* ─── 6. Model Comparison & Validation Benchmark ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
                Validation-Driven Model Benchmark
              </h3>
              <span style={{ background: `${COLORS.success}22`, color: COLORS.success, border: `1px solid ${COLORS.success}44`, borderRadius: 6, padding: '2px 8px', fontSize: 10, fontWeight: 700 }}>
                Selected on Validation Set
              </span>
            </div>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '4px 0 0' }}>
              Model recommendation is determined strictly by validation performance to ensure unbiased evaluation.
            </p>
          </div>
          <button
            onClick={handleCompareModels}
            disabled={compareLoading}
            style={{
              background: COLORS.bg.dark,
              border: `1px solid ${COLORS.bg.border}`,
              color: COLORS.primary,
              borderRadius: 8,
              padding: '6px 14px',
              fontSize: 12,
              fontWeight: 600,
              cursor: compareLoading ? 'not-allowed' : 'pointer',
            }}
          >
            {compareLoading ? 'Benchmarking...' : 'Refresh Benchmark'}
          </button>
        </div>

        {compareError && (
          <div style={{ color: COLORS.accent, fontSize: 13, marginBottom: 12 }}>⚠️ {compareError}</div>
        )}

        {compareData?.ranking ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Validation RMSE Bar Chart */}
            <div style={{ background: COLORS.bg.dark, borderRadius: 12, padding: '16px 20px', border: `1px solid ${COLORS.bg.border}` }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginBottom: 2 }}>
                Validation RMSE by Candidate Model (Lower is Better)
              </div>
              <div style={{ fontSize: 11, color: COLORS.text.muted, marginBottom: 12 }}>
                Validation split RMSE used for adaptive model selection
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={compareData.ranking} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                  <XAxis dataKey="model" tick={{ fill: COLORS.text.primary, fontSize: 11 }} tickFormatter={(m) => m.replace('_', ' ').toUpperCase()} />
                  <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                  <Tooltip contentStyle={customTooltipStyle} formatter={(val) => [formatNumber(val), 'Val RMSE']} />
                  <Bar dataKey="validation_metrics.rmse" name="Val RMSE" radius={[4, 4, 0, 0]}>
                    {compareData.ranking.map((entry) => (
                      <Cell
                        key={entry.model}
                        fill={entry.model === compareData.recommended_model ? COLORS.primary : COLORS.secondary}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Ranking Table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${COLORS.bg.border}`, color: COLORS.text.muted }}>
                    <th style={{ padding: '10px 12px' }}>Rank</th>
                    <th style={{ padding: '10px 12px' }}>Model</th>
                    <th style={{ padding: '10px 12px' }}>Val RMSE</th>
                    <th style={{ padding: '10px 12px' }}>Val MAE</th>
                    <th style={{ padding: '10px 12px' }}>Val MAPE</th>
                    <th style={{ padding: '10px 12px' }}>Val R²</th>
                    <th style={{ padding: '10px 12px' }}>Test RMSE</th>
                  </tr>
                </thead>
                <tbody>
                  {compareData.ranking.map((row) => {
                    const isRec = row.model === compareData.recommended_model;
                    return (
                      <tr
                        key={row.model}
                        style={{
                          borderBottom: `1px solid ${COLORS.bg.border}33`,
                          background: isRec ? `${COLORS.primary}12` : 'transparent',
                        }}
                      >
                        <td style={{ padding: '12px', fontWeight: 700, color: isRec ? COLORS.primary : COLORS.text.primary }}>
                          #{row.rank}
                        </td>
                        <td style={{ padding: '12px', fontWeight: 600, color: COLORS.text.primary, textTransform: 'capitalize' }}>
                          {row.model.replace('_', ' ')}
                          {isRec && (
                            <span
                              style={{
                                marginLeft: 8,
                                background: `${COLORS.success}22`,
                                color: COLORS.success,
                                border: `1px solid ${COLORS.success}44`,
                                borderRadius: 6,
                                padding: '2px 8px',
                                fontSize: 10,
                                fontWeight: 700,
                              }}
                            >
                              RECOMMENDED WINNER
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '12px', fontFamily: "'Space Mono', monospace", color: COLORS.text.primary, fontWeight: 700 }}>
                          {formatNumber(row.validation_metrics?.rmse)}
                        </td>
                        <td style={{ padding: '12px', fontFamily: "'Space Mono', monospace", color: COLORS.text.secondary }}>
                          {formatNumber(row.validation_metrics?.mae)}
                        </td>
                        <td style={{ padding: '12px', fontFamily: "'Space Mono', monospace", color: COLORS.text.secondary }}>
                          {formatMape(row.validation_metrics?.mape)}
                        </td>
                        <td style={{ padding: '12px', fontFamily: "'Space Mono', monospace", color: COLORS.text.secondary }}>
                          {formatNumber(row.validation_metrics?.r2, 3)}
                        </td>
                        <td style={{ padding: '12px', fontFamily: "'Space Mono', monospace", color: COLORS.text.muted }}>
                          {formatNumber(row.test_metrics?.rmse)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div style={{ padding: '20px 0', color: COLORS.text.muted, fontSize: 13 }}>
            Click "Compare Models" above to run validation benchmark across all candidate models for {domain}.
          </div>
        )}
      </div>

      {/* ─── 7. Explainable AI (Top Forecast Drivers) ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Top Forecast Drivers (Explainable AI — SHAP)
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '2px 0 0' }}>
              Mean |SHAP Value| — Higher values indicate features with greater influence on predictions.
            </p>
          </div>
          {isExplainableModel && (
            <button
              onClick={() => fetchExplainability(domain, forecastData?.model || model)}
              disabled={explainLoading}
              style={{
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                color: COLORS.primary,
                borderRadius: 8,
                padding: '6px 14px',
                fontSize: 12,
                fontWeight: 600,
                cursor: explainLoading ? 'not-allowed' : 'pointer',
              }}
            >
              {explainLoading ? 'Computing SHAP...' : 'Recompute SHAP'}
            </button>
          )}
        </div>

        {!isExplainableModel ? (
          <div style={{ padding: '20px', background: COLORS.bg.dark, borderRadius: 12, border: `1px solid ${COLORS.bg.border}`, color: COLORS.text.secondary, fontSize: 13 }}>
            ℹ️ Feature explainability via Tree SHAP is available for <span style={{ color: COLORS.primary, fontWeight: 700 }}>Random Forest</span> and <span style={{ color: COLORS.primary, fontWeight: 700 }}>XGBoost</span> tree ensemble models. Switch to one of these models to inspect top feature drivers.
          </div>
        ) : explainLoading ? (
          <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: COLORS.text.secondary }}>
            Computing SHAP values across evaluation dataset...
          </div>
        ) : explainError ? (
          <div style={{ color: COLORS.accent, fontSize: 13 }}>⚠️ {explainError}</div>
        ) : explainData ? (() => {
          const featureImportances = (
            explainData.feature_importance
              ? Object.entries(explainData.feature_importance).map(([feature, importance]) => ({
                  feature: formatFeatureName(feature),
                  rawFeature: feature,
                  importance: Number(importance),
                }))
              : (explainData.feature_importances || []).map((item) => ({
                  feature: formatFeatureName(item.feature),
                  rawFeature: item.feature,
                  importance: Number(item.importance),
                }))
          );

          if (featureImportances.length === 0) {
            return (
              <div style={{ padding: '20px 0', color: COLORS.text.muted, fontSize: 13 }}>
                No feature importances returned for model {explainData.model}.
              </div>
            );
          }

          return (
            <div>
              <div style={{ fontSize: 13, color: COLORS.text.secondary, marginBottom: 16 }}>
                Top feature drivers for model <span style={{ color: COLORS.primary, fontWeight: 700, textTransform: 'capitalize' }}>{explainData.model?.replace('_', ' ')}</span>:
              </div>
              <ResponsiveContainer width="100%" height={Math.max(240, featureImportances.slice(0, 10).length * 32)}>
                <BarChart
                  layout="vertical"
                  data={featureImportances.slice(0, 10)}
                  margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                  <XAxis type="number" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: COLORS.text.primary, fontSize: 11 }}
                    width={130}
                  />
                  <Tooltip contentStyle={customTooltipStyle} formatter={(val) => [formatNumber(val, 4), 'Mean |SHAP|']} />
                  <Bar dataKey="importance" fill={COLORS.primary} radius={[0, 4, 4, 0]}>
                    {featureImportances.slice(0, 10).map((_, i) => (
                      <Cell key={i} fill={i < 3 ? COLORS.primary : COLORS.secondary} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          );
        })() : (
          <div style={{ padding: '20px 0', color: COLORS.text.muted, fontSize: 13 }}>
            Click "Recompute SHAP" to view feature importance analysis.
          </div>
        )}
      </div>
    </div>
  );
}
