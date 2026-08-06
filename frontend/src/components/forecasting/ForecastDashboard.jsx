import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { runForecast, compareModels, explainForecast, DOMAIN_FEATURE_ARTIFACTS } from '../../services/mlApi';

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
  padding: '10px 14px',
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

export default function ForecastDashboard({ initialDomain = 'retail' }) {
  const [domain, setDomain] = useState(initialDomain);
  const [model, setModel] = useState('random_forest');

  // Sync state if initialDomain prop changes
  useEffect(() => {
    setDomain(initialDomain);
    handleRunForecast('random_forest', initialDomain);
  }, [initialDomain]);
  
  // Forecast State
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [forecastError, setForecastError] = useState(null);

  // Compare State
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData, setCompareData] = useState(null);
  const [compareError, setCompareError] = useState(null);

  // Auto Select recommended model note
  const [autoSelectedModel, setAutoSelectedModel] = useState(null);

  // Explainability State
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainData, setExplainData] = useState(null);
  const [explainError, setExplainError] = useState(null);

  // Auto run initial forecast on mount
  useEffect(() => {
    handleRunForecast('random_forest', 'retail');
  }, []);

  // Primary forecast handler
  const handleRunForecast = async (targetModel = model, targetDomain = domain) => {
    setForecastLoading(true);
    setForecastError(null);
    setExplainData(null);
    setExplainError(null);

    let actualModel = targetModel;
    let recModel = null;

    try {
      // Handle Auto Select
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

      // Trigger SHAP explainability for supported tree models (RF & XGBoost)
      if (['random_forest', 'xgboost'].includes(actualModel)) {
        fetchExplainability(targetDomain, actualModel);
      }
    } catch (err) {
      console.error('Forecast error:', err);
      setForecastError(err.message || 'Failed to generate forecast');
    } finally {
      setForecastLoading(false);
    }
  };

  // Model comparison handler
  const handleCompareModels = async () => {
    setCompareLoading(true);
    setCompareError(null);
    try {
      const res = await compareModels({ domain });
      setCompareData(res);
    } catch (err) {
      console.error('Comparison error:', err);
      setCompareError(err.message || 'Failed to compare models');
    } finally {
      setCompareLoading(false);
    }
  };

  // Explainability handler
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
      setExplainError(err.message || 'Explainability request failed');
    } finally {
      setExplainLoading(false);
    }
  };

  // Prepare chart series from predictions[]
  const chartData = forecastData?.predictions?.map((item) => ({
    rawTimestamp: item.timestamp,
    timestamp: formatTimestamp(item.timestamp),
    actual: item.actual !== undefined && item.actual !== null ? Number(item.actual) : null,
    predicted: item.predicted !== undefined && item.predicted !== null ? Number(item.predicted) : null,
  })) || [];

  const activeModelDisplay = autoSelectedModel
    ? `${autoSelectedModel} (Auto Selected)`
    : forecastData?.model || model;

  const isExplainableModel = ['random_forest', 'xgboost'].includes(forecastData?.model || model);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* ─── Header & Control Bar ─── */}
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
              Forecast Intelligence Dashboard
            </h1>
            <p style={{ fontSize: 13, color: COLORS.text.secondary, margin: '4px 0 0' }}>
              Interactive ML demand forecasting with validation-driven adaptive model selection & feature explainability
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
              {compareLoading ? 'Evaluating...' : '📊 Compare Models'}
            </button>
          </div>
        </div>

        {/* Form Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
          {/* Domain Selection */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Target Domain
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              {[
                { id: 'retail', label: '🛒 Retail', sub: 'Sales Demand' },
                { id: 'electricity', label: '⚡ Electricity', sub: 'System Load' },
              ].map((d) => (
                <button
                  key={d.id}
                  onClick={() => {
                    setDomain(d.id);
                    handleRunForecast(model, d.id);
                  }}
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

          {/* Model Selection */}
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
              <option value="hybrid">Hybrid (Validation-Weighted Ensemble)</option>
              <option value="auto_select">⚡ Auto Select (Validation Recommended)</option>
              {domain === 'electricity' && <option value="arima">ARIMA (Time Series)</option>}
            </select>
          </div>

          {/* Metadata Display */}
          <div style={{ background: COLORS.bg.dark, borderRadius: 10, padding: '12px 16px', border: `1px solid ${COLORS.bg.border}`, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Active Configuration</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginTop: 4, fontFamily: "'Space Mono', monospace" }}>
              {domain.toUpperCase()} • {activeModelDisplay.toUpperCase()}
            </div>
            {forecastData?.target && (
              <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2 }}>
                Target Variable: <span style={{ color: COLORS.primary }}>{forecastData.target}</span>
              </div>
            )}
          </div>
        </div>

        {/* Auto Select / Recommended Banner */}
        {autoSelectedModel && (
          <div style={{ background: `${COLORS.success}15`, border: `1px solid ${COLORS.success}44`, borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 20 }}>🏆</span>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.success }}>
                Auto Select Recommendation: <span style={{ textTransform: 'capitalize' }}>{autoSelectedModel.replace('_', ' ')}</span>
              </div>
              <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2 }}>
                Selected using optimal validation RMSE performance across candidate models.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Alert */}
      {forecastError && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: '16px 20px', color: COLORS.accent, fontSize: 14, fontWeight: 600 }}>
          ⚠️ Forecast Error: {forecastError}
        </div>
      )}

      {/* ─── P0: Metric Cards ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
        {[
          {
            label: 'MAE',
            title: 'Mean Absolute Error',
            value: formatNumber(forecastData?.test_metrics?.mae),
            sub: 'Test Set Evaluation',
            color: COLORS.primary,
          },
          {
            label: 'RMSE',
            title: 'Root Mean Squared Error',
            value: formatNumber(forecastData?.test_metrics?.rmse),
            sub: 'Test Set Evaluation',
            color: COLORS.secondary,
          },
          {
            label: 'MAPE',
            title: 'Mean Abs % Error',
            value: formatMape(forecastData?.test_metrics?.mape),
            sub: 'Test Set Percentage',
            color: COLORS.success,
          },
          {
            label: 'R² Score',
            title: 'Coefficient of Determination',
            value: formatNumber(forecastData?.test_metrics?.r2, 4),
            sub: 'Test Set Fit Quality',
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
            }}
          >
            <div style={{ fontSize: 12, color: COLORS.text.secondary, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {m.label}
            </div>
            <div style={{ fontSize: 26, fontWeight: 800, color: m.color, fontFamily: "'Space Mono', monospace", margin: '6px 0' }}>
              {forecastLoading ? '...' : m.value}
            </div>
            <div style={{ fontSize: 11, color: COLORS.text.muted }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* Hybrid Component Weights (if applicable) */}
      {forecastData?.model === 'hybrid' && forecastData?.weights && (
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 12, padding: '14px 20px', display: 'flex', alignItems: 'center', gap: 20 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary }}>Ensemble Component Weights:</span>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {Object.entries(forecastData.weights).map(([comp, w]) => (
              <span key={comp} style={{ fontSize: 13, color: COLORS.primary, fontFamily: "'Space Mono', monospace", background: COLORS.bg.dark, padding: '4px 10px', borderRadius: 6, border: `1px solid ${COLORS.bg.border}` }}>
                {comp}: {(Number(w) * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ─── P0: Actual vs Predicted Line Chart ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Actual vs Predicted Demand Series
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '2px 0 0' }}>
              Test set forecasting evaluation ({chartData.length} predictions)
            </p>
          </div>
          {forecastData?.total_test_predictions && (
            <span style={{ fontSize: 12, color: COLORS.text.muted, fontFamily: "'Space Mono', monospace" }}>
              Showing {chartData.length} of {forecastData.total_test_predictions} total test samples
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
              <Tooltip contentStyle={customTooltipStyle} formatter={(val) => [formatNumber(val), 'Value']} />
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
          <div style={{ height: 260, display: 'flex', alignItems: 'center', justifyContent: 'center', color: COLORS.text.muted }}>
            No forecast data available. Click "Run Forecast" above.
          </div>
        )}
      </div>

      {/* ─── P1: Model Comparison Section ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Validation-Driven Model Benchmark & Adaptive Selection
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '2px 0 0' }}>
              Model selection is performed strictly on validation performance to prevent test-set leakage
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
                        background: isRec ? `${COLORS.primary}10` : 'transparent',
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
                            RECOMMENDED
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '12px', fontFamily: "'Space Mono', monospace", color: COLORS.text.primary }}>
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
            <div style={{ marginTop: 12, fontSize: 11, color: COLORS.text.muted, fontStyle: 'italic' }}>
              Note: Test set metrics are reported separately for final evaluation only and were not used during model recommendation.
            </div>
          </div>
        ) : (
          <div style={{ padding: '20px 0', textTransform: 'none', color: COLORS.text.muted, fontSize: 13 }}>
            Click "Compare Models" above to run validation benchmark across all candidate models for {domain}.
          </div>
        )}
      </div>

      {/* ─── P2: Explainability Section (Global SHAP) ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Explainable AI — Feature Importance (SHAP)
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '2px 0 0' }}>
              Global feature attributions for model predictions
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
            ℹ️ Explainability via Tree SHAP is available for <span style={{ color: COLORS.primary, fontWeight: 700 }}>Random Forest</span> and <span style={{ color: COLORS.primary, fontWeight: 700 }}>XGBoost</span> models. Switch to one of these models to inspect feature drivers.
          </div>
        ) : explainLoading ? (
          <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: COLORS.text.secondary }}>
            Computing SHAP values across evaluation set...
          </div>
        ) : explainError ? (
          <div style={{ color: COLORS.accent, fontSize: 13 }}>⚠️ {explainError}</div>
        ) : explainData?.feature_importances ? (
          <div>
            <div style={{ fontSize: 13, color: COLORS.text.secondary, marginBottom: 16 }}>
              Top feature drivers for model <span style={{ color: COLORS.primary, fontWeight: 700 }}>{explainData.model}</span>:
            </div>
            <ResponsiveContainer width="100%" height={Math.max(220, explainData.feature_importances.slice(0, 10).length * 28)}>
              <BarChart
                layout="vertical"
                data={explainData.feature_importances.slice(0, 10)}
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                <XAxis type="number" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fill: COLORS.text.primary, fontSize: 12 }}
                  width={110}
                />
                <Tooltip contentStyle={customTooltipStyle} formatter={(val) => [val.toFixed(4), 'Mean |SHAP|']} />
                <Bar dataKey="importance" fill={COLORS.primary} radius={[0, 4, 4, 0]}>
                  {explainData.feature_importances.slice(0, 10).map((_, i) => (
                    <Cell key={i} fill={i < 3 ? COLORS.primary : COLORS.secondary} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div style={{ padding: '20px 0', color: COLORS.text.muted, fontSize: 13 }}>
            Click "Recompute SHAP" to view feature importance analysis.
          </div>
        )}
      </div>
    </div>
  );
}
