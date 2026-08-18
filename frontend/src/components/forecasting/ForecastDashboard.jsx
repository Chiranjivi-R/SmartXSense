import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  runForecast,
  compareModels,
  explainForecast,
  runFutureForecast,
  fetchForecastHistory,
  fetchDatasets,
  FUTURE_HORIZON_PRESETS,
  FUTURE_SUPPORTED_MODELS,
  DOMAIN_FEATURE_ARTIFACTS,
  uploadDataset,
  getDatasetDetail,
  validateDataset,
  preprocessDataset,
} from '../../services/mlApi';
import {
  exportFutureForecastCsv,
  exportFutureForecastJson,
  exportTestForecastCsv,
  exportTestForecastJson,
} from '../../utils/forecastExport';

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

export default function ForecastDashboard({ initialDomain = 'retail', userRole = 'viewer', accessToken = null }) {
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

  const [futureModel, setFutureModel] = useState('random_forest');
  const [futureHorizon, setFutureHorizon] = useState(
    FUTURE_HORIZON_PRESETS[initialDomain]?.[0] || 7
  );
  const [futureLoading, setFutureLoading] = useState(false);
  const [futureData, setFutureData] = useState(null);
  const [futureError, setFutureError] = useState(null);

  const [customFeaturesDir, setCustomFeaturesDir] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadDomain, setUploadDomain] = useState(initialDomain);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadInfo, setUploadInfo] = useState(null);
  const [uploadValidation, setUploadValidation] = useState(null);
  const [uploadProcess, setUploadProcess] = useState(null);

  // Persistent Dataset Registry State
  const [datasetsList, setDatasetsList] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState(null);

  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [exportStatus, setExportStatus] = useState(null);
  const canManageDatasets = ['admin', 'analyst', 'data_scientist'].includes(String(userRole).toLowerCase());
  const domainRequestToken = useRef(0);

  const loadDatasets = async (targetDomain = domain, autoSelectId = null) => {
    setDatasetsLoading(true);
    setDatasetsError(null);
    try {
      const res = await fetchDatasets({ domain: targetDomain, accessToken });
      const list = Array.isArray(res?.datasets) ? res.datasets : [];
      setDatasetsList(list);

      let nextSelectedId = autoSelectId;
      if (!nextSelectedId && selectedDatasetId && list.some((d) => d.dataset_id === selectedDatasetId)) {
        nextSelectedId = selectedDatasetId;
      }
      if (!nextSelectedId && list.length > 0) {
        nextSelectedId = list[0].dataset_id;
      }
      setSelectedDatasetId(nextSelectedId || null);
      return list;
    } catch (err) {
      console.error('Failed to load dataset registry:', err);
      setDatasetsError(err.message || 'Unable to load dataset registry.');
      setDatasetsList([]);
      return [];
    } finally {
      setDatasetsLoading(false);
    }
  };

  const getActiveFeaturesDir = (override) => {
    if (override !== undefined) return override;
    const dsObj = datasetsList.find((d) => d.dataset_id === selectedDatasetId);
    if (dsObj) {
      return dsObj.features_dir || dsObj.dataset_id;
    }
    return customFeaturesDir || undefined;
  };

  const handleExport = (kind, format) => {
    setExportStatus(null);
    try {
      let filename;
      if (kind === 'test') {
        if (!forecastData) throw new Error('Run a forecast before exporting.');
        filename = format === 'csv'
          ? exportTestForecastCsv(forecastData)
          : exportTestForecastJson(forecastData);
      } else {
        if (!futureData) throw new Error('Run a future forecast before exporting.');
        filename = format === 'csv'
          ? exportFutureForecastCsv(futureData)
          : exportFutureForecastJson(futureData);
      }
      setExportStatus(`Downloaded ${filename}`);
    } catch (err) {
      console.error('Export error:', err);
      setExportStatus(err.message || 'Export failed.');
    }
  };

  const loadForecastHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetchForecastHistory({ limit: 20 });
      setHistoryRuns(Array.isArray(res?.runs) ? res.runs : []);
    } catch (err) {
      console.error('History load error:', err);
      setHistoryError(err.message || 'Unable to load forecast history.');
      setHistoryRuns([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  // Sync state if initialDomain prop changes
  useEffect(() => {
    handleDomainSwitch(initialDomain);
  }, [initialDomain]);

  // Initial forecast on mount
  useEffect(() => {
    const init = async () => {
      const list = await loadDatasets(initialDomain);
      const defaultDs = list[0];
      const targetFeaturesDir = defaultDs?.features_dir || defaultDs?.dataset_id;
      handleRunForecast('random_forest', initialDomain, targetFeaturesDir);
      loadForecastHistory();
    };
    init();
  }, []);

  // Clear stale state when domain changes
  const handleDomainSwitch = async (newDomain) => {
    if (!['retail', 'electricity'].includes(newDomain)) return;
    domainRequestToken.current += 1;
    const validModel = newDomain === 'retail' && model === 'arima'
      ? 'random_forest'
      : model;
    setDomain(newDomain);
    setModel(validModel);
    setUploadDomain(newDomain);
    setForecastData(null);
    setCompareData(null);
    setExplainData(null);
    setAutoSelectedModel(null);
    setForecastError(null);
    setCompareError(null);
    setExplainError(null);
    setFutureData(null);
    setFutureError(null);
    setFutureModel('random_forest');
    setFutureHorizon(FUTURE_HORIZON_PRESETS[newDomain]?.[0] || 7);
    setCustomFeaturesDir(null);

    const list = await loadDatasets(newDomain);
    const defaultDs = list[0];
    const targetFeaturesDir = defaultDs?.features_dir || defaultDs?.dataset_id;
    handleRunForecast(validModel, newDomain, targetFeaturesDir);
  };

  // Primary forecast runner
  const handleRunForecast = async (
    targetModel = model,
    targetDomain = domain,
    featuresDirOverride = undefined,
    forceRetrain = false,
  ) => {
    const requestToken = domainRequestToken.current;
    setForecastLoading(true);
    setForecastError(null);
    setExplainData(null);
    setExplainError(null);

    let actualModel = targetModel;
    let recModel = null;
    const activeFeaturesDir = getActiveFeaturesDir(featuresDirOverride);

    try {
      // Auto Select workflow
      if (targetModel === 'auto_select') {
        setCompareLoading(true);
        try {
          const compResult = await compareModels({
            domain: targetDomain,
            featuresDir: activeFeaturesDir,
          });
          if (requestToken !== domainRequestToken.current) return;
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
        featuresDir: activeFeaturesDir,
        predictionLimit: 100,
        selectionMethod: targetModel === 'auto_select' ? 'validation_rmse' : undefined,
        forceRetrain,
      });

      if (requestToken !== domainRequestToken.current) return;
      setForecastData(res);
      loadForecastHistory();

      // Trigger SHAP explainability for supported tree models
      if (['random_forest', 'xgboost'].includes(actualModel)) {
        fetchExplainability(targetDomain, actualModel, activeFeaturesDir);
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
    const requestToken = domainRequestToken.current;
    setCompareLoading(true);
    setCompareError(null);
    try {
      const res = await compareModels({
        domain,
        featuresDir: getActiveFeaturesDir(),
      });
      if (requestToken === domainRequestToken.current) setCompareData(res);
    } catch (err) {
      console.error('Comparison error:', err);
      setCompareError('Model comparison request failed. Please verify backend service.');
    } finally {
      setCompareLoading(false);
    }
  };

  // Explainability runner
  const fetchExplainability = async (expDomain, expModel, featuresDirOverride = undefined) => {
    const requestToken = domainRequestToken.current;
    setExplainLoading(true);
    setExplainError(null);
    try {
      const res = await explainForecast({
        domain: expDomain,
        model: expModel,
        featuresDir: getActiveFeaturesDir(featuresDirOverride),
        explanationType: 'global',
        method: 'shap',
      });
      if (requestToken === domainRequestToken.current) setExplainData(res);
    } catch (err) {
      console.error('Explainability error:', err);
      setExplainError('Explainability request failed for this model.');
    } finally {
      setExplainLoading(false);
    }
  };

  const futureSupportedModels = FUTURE_SUPPORTED_MODELS[domain] || [];
  const isFutureModelSupported = futureSupportedModels.includes(futureModel);

  const handleRunFutureForecast = async () => {
    const requestToken = domainRequestToken.current;
    if (!isFutureModelSupported) {
      setFutureError(`Future forecasting is not currently supported for model ${futureModel}.`);
      return;
    }
    setFutureLoading(true);
    setFutureError(null);
    try {
      const res = await runFutureForecast({
        domain,
        model: futureModel,
        horizon: Number(futureHorizon),
        featuresDir: getActiveFeaturesDir(),
      });
      if (requestToken !== domainRequestToken.current) return;
      setFutureData(res);
      loadForecastHistory();
    } catch (err) {
      console.error('Future forecast error:', err);
      setFutureError(err.message || 'Future forecast request failed. Please verify backend service.');
    } finally {
      setFutureLoading(false);
    }
  };

  const handleUseDemoDataset = () => {
    setCustomFeaturesDir(null);
    setUploadInfo(null);
    setUploadValidation(null);
    setUploadProcess(null);
    setUploadError(null);
    setUploadFile(null);
    const demoId = domain === 'retail' ? 'demo_retail' : 'demo_electricity';
    setSelectedDatasetId(demoId);
    handleRunForecast(model, domain, demoId);
  };

  const handleUploadAndPrepare = async () => {
    if (!uploadFile) {
      setUploadError('Please select a CSV or Excel file to upload.');
      return;
    }
    setUploadLoading(true);
    setUploadError(null);
    setUploadInfo(null);
    setUploadValidation(null);
    setUploadProcess(null);
    try {
      if (!accessToken) throw new Error('Please sign in with a Data Scientist or Admin account to upload a dataset.');
      const token = accessToken;
      const uploaded = await uploadDataset({
        file: uploadFile,
        domain: uploadDomain,
        accessToken: token,
      });
      const detail = await getDatasetDetail({
        datasetId: uploaded.dataset_id,
        accessToken: token,
        previewRows: 5,
      });
      setUploadInfo({ ...uploaded, columns: detail.columns, preview: detail.preview });

      const validation = await validateDataset({
        datasetId: uploaded.dataset_id,
        domain: uploadDomain,
        accessToken: token,
      });
      setUploadValidation(validation);

      const processed = await preprocessDataset({
        datasetId: uploaded.dataset_id,
        accessToken: token,
      });
      setUploadProcess(processed);

      if (!processed.features_ready || !processed.features_dir) {
        throw new Error(
          processed.features_error
            || 'Upload cleaned successfully, but forecasting feature splits could not be generated. Use a longer time series.',
        );
      }

      setCustomFeaturesDir(processed.features_dir);
      setDomain(uploadDomain);
      setFutureHorizon(FUTURE_HORIZON_PRESETS[uploadDomain]?.[0] || 7);
      const list = await loadDatasets(uploadDomain, uploaded.dataset_id);
      const newDs = list.find((d) => d.dataset_id === uploaded.dataset_id);
      const targetFDir = newDs?.features_dir || processed.features_dir;
      await handleRunForecast(model, uploadDomain, targetFDir);
    } catch (err) {
      console.error('Upload pipeline error:', err);
      setUploadError(err.message || 'Dataset upload/processing failed.');
    } finally {
      setUploadLoading(false);
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

  const futureChartData = futureData?.predictions?.map((item) => ({
    rawTimestamp: item.timestamp,
    timestamp: formatTimestamp(item.timestamp),
    predicted: item.predicted !== undefined && item.predicted !== null ? Number(item.predicted) : null,
  })) || [];

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
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
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
              onClick={() => handleRunForecast(model, domain, undefined, true)}
              disabled={forecastLoading || compareLoading}
              title="Force fresh model retraining and update artifact"
              style={{
                background: 'transparent',
                color: COLORS.text.secondary,
                border: `1px solid ${COLORS.bg.border}`,
                borderRadius: 10,
                padding: '10px 14px',
                fontSize: 13,
                fontWeight: 600,
                cursor: forecastLoading || compareLoading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
              }}
            >
              🔄 Retrain
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

          {/* Dataset Selector */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Dataset {datasetsLoading && '(Loading...)'}
            </label>
            <select
              value={selectedDatasetId || ''}
              onChange={(e) => {
                const selectedId = e.target.value;
                setSelectedDatasetId(selectedId);
                const dsObj = datasetsList.find((d) => d.dataset_id === selectedId);
                const fDir = dsObj?.features_dir || dsObj?.dataset_id;
                handleRunForecast(model, domain, fDir);
              }}
              disabled={datasetsLoading || datasetsList.length === 0}
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
              {datasetsList.length === 0 ? (
                <option value="">No datasets available</option>
              ) : (
                datasetsList.map((d) => (
                  <option key={d.dataset_id} value={d.dataset_id}>
                    {d.filename || d.dataset_id} {d.is_demo ? '(Demo)' : '(Uploaded)'}
                  </option>
                ))
              )}
            </select>
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
            <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Dataset & Artifact Status</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginTop: 4, fontFamily: "'Space Mono', monospace" }}>
              {(() => {
                const activeDs = datasetsList.find((d) => d.dataset_id === selectedDatasetId);
                if (activeDs) {
                  return `${activeDs.filename} ${activeDs.is_demo ? '(Demo)' : '(Uploaded)'}`;
                }
                return customFeaturesDir ? 'Uploaded Dataset' : `Demo Dataset: ${domain.toUpperCase()}`;
              })()}
            </div>
            <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Active Model: <span style={{ color: COLORS.primary, fontWeight: 700, textTransform: 'capitalize' }}>{activeModelDisplay}</span></span>
              {forecastData?.execution && (
                <span style={{ fontSize: 11, color: forecastData.execution.artifact_reused ? COLORS.success : COLORS.warning, fontWeight: 700 }}>
                  {forecastData.execution.artifact_reused ? '⚡ Artifact Reused' : '🏋️ Model Trained'} ({forecastData.execution.execution_time_ms} ms)
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Dataset Upload (optional — demo datasets remain default) */}
        {canManageDatasets && <div style={{ marginTop: 4, background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 12, padding: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary }}>Dataset Upload</div>
              <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2 }}>
                Optional CSV/Excel upload using the existing preprocessing + feature-engineering pipeline. Demo datasets stay available.
              </div>
            </div>
            <button
              onClick={handleUseDemoDataset}
              style={{
                background: 'transparent',
                border: `1px solid ${COLORS.bg.border}`,
                color: COLORS.text.secondary,
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 12,
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              Use Demo Dataset
            </button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 6 }}>FILE</label>
              <input
                type="file"
                accept=".csv,.xlsx"
                onChange={(e) => {
                  setUploadFile(e.target.files?.[0] || null);
                  setUploadError(null);
                }}
                style={{ width: '100%', color: COLORS.text.secondary, fontSize: 12 }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 6 }}>DOMAIN</label>
              <select
                value={uploadDomain}
                onChange={(e) => setUploadDomain(e.target.value)}
                style={{
                  width: '100%',
                  background: COLORS.bg.card,
                  border: `1px solid ${COLORS.bg.border}`,
                  borderRadius: 8,
                  padding: '8px 10px',
                  color: COLORS.text.primary,
                  fontSize: 13,
                }}
              >
                <option value="retail">Retail</option>
                <option value="electricity">Electricity</option>
              </select>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button
                onClick={handleUploadAndPrepare}
                disabled={uploadLoading}
                style={{
                  width: '100%',
                  background: uploadLoading ? COLORS.bg.border : `linear-gradient(135deg, ${COLORS.success}, ${COLORS.primary})`,
                  color: '#fff',
                  border: 'none',
                  borderRadius: 8,
                  padding: '10px 14px',
                  fontWeight: 700,
                  cursor: uploadLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {uploadLoading ? 'Uploading & Preparing...' : 'Upload → Validate → Prepare'}
              </button>
            </div>
          </div>
          {uploadFile && (
            <div style={{ marginTop: 10, fontSize: 12, color: COLORS.text.secondary }}>
              Selected: <strong style={{ color: COLORS.text.primary }}>{uploadFile.name}</strong>
              {' · '}{(uploadFile.size / 1024).toFixed(1)} KB
            </div>
          )}
          {uploadError && (
            <div style={{ marginTop: 10, color: COLORS.accent, fontSize: 13 }}>⚠️ {uploadError}</div>
          )}
          {uploadInfo && (
            <div style={{ marginTop: 10, fontSize: 12, color: COLORS.text.secondary, lineHeight: 1.6 }}>
              Uploaded <strong style={{ color: COLORS.text.primary }}>{uploadInfo.filename}</strong>
              {' · '}rows {uploadInfo.row_count}
              {' · '}cols {uploadInfo.column_count}
              {uploadInfo.columns?.length ? ` · columns: ${uploadInfo.columns.slice(0, 8).join(', ')}${uploadInfo.columns.length > 8 ? '…' : ''}` : ''}
              {uploadValidation ? ` · validation: ${uploadValidation.is_valid ? 'passed' : 'failed'}` : ''}
              {uploadProcess ? ` · features_ready: ${uploadProcess.features_ready ? 'yes' : 'no'}` : ''}
              {customFeaturesDir ? ` · active features_dir set` : ''}
            </div>
          )}
        </div>}

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
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, color: COLORS.primary, padding: '4px 12px', borderRadius: 8, fontSize: 12, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>
              Model: {forecastData.model?.replace('_', ' ').toUpperCase()}
            </span>
            <span style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, color: COLORS.success, padding: '4px 12px', borderRadius: 8, fontSize: 12, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>
              {forecastData.prediction_count || chartData.length} Forecast Points
            </span>
            <button
              type="button"
              onClick={() => handleExport('test', 'csv')}
              style={{
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                color: COLORS.text.primary,
                borderRadius: 8,
                padding: '6px 12px',
                fontSize: 12,
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              Export CSV
            </button>
            <button
              type="button"
              onClick={() => handleExport('test', 'json')}
              style={{
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                color: COLORS.primary,
                borderRadius: 8,
                padding: '6px 12px',
                fontSize: 12,
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              Export JSON
            </button>
          </div>
        </div>
      )}

      {exportStatus && (
        <div style={{ fontSize: 12, color: COLORS.text.secondary, padding: '0 4px' }}>
          {exportStatus}
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
          {forecastData.adaptive_selection && (
            <div style={{ marginTop: 14, padding: 12, borderRadius: 10, background: `${COLORS.primary}10`, border: `1px solid ${COLORS.primary}33`, color: COLORS.text.secondary, fontSize: 13 }}>
              <strong style={{ color: COLORS.primary }}>Validation-only adaptive decision:</strong>{' '}
              {forecastData.adaptive_selection.selection_reason}
              <div style={{ marginTop: 4 }}>Selected model: <strong style={{ color: COLORS.text.primary, textTransform: 'capitalize' }}>{forecastData.selected_model?.replace('_', ' ')}</strong>. Held-out test metrics are reported only after this decision.</div>
            </div>
          )}
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

      {/* ─── 5b. Future Forecast (no actual values) ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Future Forecast
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '4px 0 0', maxWidth: 720 }}>
              Out-of-sample demand forecast beyond the last historical timestamp. Actual values are not available.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ background: `${COLORS.warning}18`, color: COLORS.warning, border: `1px solid ${COLORS.warning}44`, borderRadius: 8, padding: '4px 10px', fontSize: 11, fontWeight: 700 }}>
              Future Forecast — Actual values are not available
            </span>
            <button
              type="button"
              onClick={() => handleExport('future', 'csv')}
              disabled={!futureData}
              style={{
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                color: futureData ? COLORS.text.primary : COLORS.text.muted,
                borderRadius: 8,
                padding: '6px 12px',
                fontSize: 12,
                fontWeight: 700,
                cursor: futureData ? 'pointer' : 'not-allowed',
              }}
            >
              Export CSV
            </button>
            <button
              type="button"
              onClick={() => handleExport('future', 'json')}
              disabled={!futureData}
              style={{
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                color: futureData ? COLORS.primary : COLORS.text.muted,
                borderRadius: 8,
                padding: '6px 12px',
                fontSize: 12,
                fontWeight: 700,
                cursor: futureData ? 'pointer' : 'not-allowed',
              }}
            >
              Export JSON
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 16 }}>
          <div>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 6, textTransform: 'uppercase' }}>
              Future Model
            </label>
            <select
              value={futureModel}
              onChange={(e) => {
                setFutureModel(e.target.value);
                setFutureError(null);
              }}
              style={{
                width: '100%',
                background: COLORS.bg.dark,
                border: `1px solid ${COLORS.bg.border}`,
                borderRadius: 10,
                padding: '10px 12px',
                color: COLORS.text.primary,
                fontSize: 13,
                fontWeight: 600,
                outline: 'none',
              }}
            >
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
              {domain === 'electricity' && <option value="arima">ARIMA</option>}
              <option value="hybrid">Hybrid (unsupported)</option>
              <option value="lstm">LSTM (unsupported)</option>
            </select>
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 6, textTransform: 'uppercase' }}>
              Horizon
            </label>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(FUTURE_HORIZON_PRESETS[domain] || [7]).map((h) => (
                <button
                  key={h}
                  onClick={() => setFutureHorizon(h)}
                  style={{
                    padding: '8px 14px',
                    borderRadius: 8,
                    border: `1px solid ${futureHorizon === h ? COLORS.primary : COLORS.bg.border}`,
                    background: futureHorizon === h ? `${COLORS.primary}22` : COLORS.bg.dark,
                    color: futureHorizon === h ? COLORS.primary : COLORS.text.secondary,
                    cursor: 'pointer',
                    fontWeight: 700,
                    fontSize: 13,
                  }}
                >
                  {h}{domain === 'electricity' ? 'h' : 'd'}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button
              onClick={handleRunFutureForecast}
              disabled={futureLoading || !isFutureModelSupported}
              style={{
                width: '100%',
                background: futureLoading || !isFutureModelSupported
                  ? COLORS.bg.border
                  : `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.accent})`,
                color: '#fff',
                border: 'none',
                borderRadius: 10,
                padding: '12px 18px',
                fontSize: 14,
                fontWeight: 700,
                cursor: futureLoading || !isFutureModelSupported ? 'not-allowed' : 'pointer',
              }}
            >
              {futureLoading ? 'Generating Future Forecast...' : 'Run Future Forecast'}
            </button>
          </div>
        </div>

        {!isFutureModelSupported && (
          <div style={{ background: `${COLORS.accent}15`, border: `1px solid ${COLORS.accent}44`, borderRadius: 10, padding: '10px 14px', color: COLORS.accent, fontSize: 13, marginBottom: 12 }}>
            Future forecasting is not currently supported for model {futureModel}. Choose Random Forest or XGBoost{domain === 'electricity' ? ' (or ARIMA)' : ''}.
          </div>
        )}

        {futureError && (
          <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 10, padding: '12px 14px', color: COLORS.accent, fontSize: 13, marginBottom: 12 }}>
            ⚠️ {futureError}
          </div>
        )}

        {futureData && (
          <div style={{ fontSize: 12, color: COLORS.text.secondary, marginBottom: 12 }}>
            Horizon: <strong style={{ color: COLORS.text.primary }}>{futureData.horizon}</strong>
            {' · '}Last historical: <strong style={{ color: COLORS.text.primary }}>{formatTimestamp(futureData.last_historical_timestamp)}</strong>
            {' · '}Policy: <strong style={{ color: COLORS.text.primary }}>{futureData.exogenous_policy}</strong>
            {futureData.artifact?.source && (
              <>
                {' · '}Artifact: <strong style={{ color: COLORS.primary }}>{futureData.artifact.source}</strong>
              </>
            )}
          </div>
        )}

        {futureLoading ? (
          <div style={{ height: 260, display: 'flex', alignItems: 'center', justifyContent: 'center', color: COLORS.text.secondary }}>
            Generating recursive future predictions...
          </div>
        ) : futureChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={futureChartData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} opacity={0.6} />
              <XAxis dataKey="timestamp" tick={{ fill: COLORS.text.muted, fontSize: 11 }} interval="preserveStartEnd" minTickGap={20} />
              <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} domain={['auto', 'auto']} />
              <Tooltip contentStyle={customTooltipStyle} formatter={(val) => [formatNumber(val), 'Predicted']} />
              <Legend wrapperStyle={{ fontSize: 13, color: COLORS.text.secondary, paddingTop: 10 }} />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Future Predicted"
                stroke={COLORS.warning}
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: COLORS.text.muted, fontSize: 13 }}>
            Select a supported model and horizon, then run a future forecast.
          </div>
        )}
      </div>

      {/* ─── 5c. Forecast History ─── */}
      <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0 }}>
              Forecast History
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '4px 0 0' }}>
              Recent successful forecast runs (metadata only — prediction arrays are not stored).
            </p>
          </div>
          <button
            onClick={loadForecastHistory}
            disabled={historyLoading}
            style={{
              background: COLORS.bg.dark,
              border: `1px solid ${COLORS.bg.border}`,
              color: COLORS.primary,
              borderRadius: 10,
              padding: '8px 14px',
              fontSize: 13,
              fontWeight: 700,
              cursor: historyLoading ? 'not-allowed' : 'pointer',
            }}
          >
            {historyLoading ? 'Refreshing...' : 'Refresh History'}
          </button>
        </div>

        {historyError && (
          <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 10, padding: '12px 14px', color: COLORS.accent, fontSize: 13, marginBottom: 12 }}>
            {historyError}
          </div>
        )}

        {!historyError && !historyLoading && historyRuns.length === 0 && (
          <div style={{ padding: '18px 0', color: COLORS.text.muted, fontSize: 13 }}>
            No forecast runs yet.
          </div>
        )}

        {historyRuns.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ color: COLORS.text.secondary, textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px', borderBottom: `1px solid ${COLORS.bg.border}`, fontWeight: 700 }}>Time</th>
                  <th style={{ padding: '8px 10px', borderBottom: `1px solid ${COLORS.bg.border}`, fontWeight: 700 }}>Domain</th>
                  <th style={{ padding: '8px 10px', borderBottom: `1px solid ${COLORS.bg.border}`, fontWeight: 700 }}>Model</th>
                  <th style={{ padding: '8px 10px', borderBottom: `1px solid ${COLORS.bg.border}`, fontWeight: 700 }}>Type</th>
                  <th style={{ padding: '8px 10px', borderBottom: `1px solid ${COLORS.bg.border}`, fontWeight: 700 }}>Metrics</th>
                </tr>
              </thead>
              <tbody>
                {historyRuns.map((run) => {
                  const ts = run.timestamp ? new Date(run.timestamp) : null;
                  const timeLabel = ts && !Number.isNaN(ts.getTime())
                    ? ts.toLocaleString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                    : '—';
                  const typeLabel = run.forecast_type === 'future'
                    ? 'Future Forecast'
                    : 'Test Evaluation';
                  const modelLabel = String(run.model || '').replace(/_/g, ' ');
                  let metricsLabel = `${run.prediction_count || 0} points`;
                  if (run.forecast_type === 'future' && run.horizon != null) {
                    metricsLabel = `Horizon ${run.horizon} · ${run.prediction_count || 0} points`;
                  } else if (run.metrics) {
                    const rmse = run.metrics.rmse != null ? `RMSE ${formatNumber(run.metrics.rmse)}` : null;
                    const mape = run.metrics.mape != null ? `MAPE ${formatMape(run.metrics.mape)}` : null;
                    metricsLabel = [rmse, mape].filter(Boolean).join(' · ') || metricsLabel;
                  }
                  return (
                    <tr key={run.run_id} style={{ color: COLORS.text.primary }}>
                      <td style={{ padding: '10px', borderBottom: `1px solid ${COLORS.bg.border}`, whiteSpace: 'nowrap' }}>{timeLabel}</td>
                      <td style={{ padding: '10px', borderBottom: `1px solid ${COLORS.bg.border}`, textTransform: 'capitalize' }}>{run.domain}</td>
                      <td style={{ padding: '10px', borderBottom: `1px solid ${COLORS.bg.border}`, textTransform: 'capitalize' }}>{modelLabel}</td>
                      <td style={{ padding: '10px', borderBottom: `1px solid ${COLORS.bg.border}` }}>{typeLabel}</td>
                      <td style={{ padding: '10px', borderBottom: `1px solid ${COLORS.bg.border}`, color: COLORS.text.secondary }}>{metricsLabel}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
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
              {compareData?.benchmark_execution && (
                <span
                  style={{
                    background: compareData.benchmark_execution.reused ? `${COLORS.primary}22` : `${COLORS.warning}22`,
                    color: compareData.benchmark_execution.reused ? COLORS.primary : COLORS.warning,
                    border: `1px solid ${compareData.benchmark_execution.reused ? COLORS.primary : COLORS.warning}44`,
                    borderRadius: 6,
                    padding: '2px 8px',
                    fontSize: 10,
                    fontWeight: 700,
                    fontFamily: "'Space Mono', monospace",
                  }}
                >
                  {compareData.benchmark_execution.reused ? '⚡ Benchmark Reused' : '🏋️ Benchmark Computed'} ({compareData.benchmark_execution.execution_time_ms} ms)
                </span>
              )}
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
