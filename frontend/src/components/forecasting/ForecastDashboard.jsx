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
  profileDataset,
  fetchDatasetGeography,
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
import {
  isApplicabilityNaError,
  parseApplicabilityDetails,
  formatUserFacingError,
} from '../../utils/applicabilityPresentation';

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
  const num = Number(val);
  const pct = num < 50 ? num * 100 : num;
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

/**
 * Clean markdown emphasis and backticks from heading text, preserving emojis and titles.
 */
function cleanHeadingText(text) {
  if (typeof text !== 'string') return text;
  return text
    .replace(/\*\*\*/g, '')
    .replace(/\*\*/g, '')
    .replace(/(^|\s)\*([^*]+)\*(\s|$)/g, '$1$2$3')
    .replace(/`([^`]+)`/g, '$1')
    .trim();
}

/**
 * Render inline markdown (bold, italic, code) into styled React elements.
 * Transforms **text** to visually emphasized bold text and *text* to italic,
 * preventing literal markdown asterisks from appearing in the UI while
 * preserving mathematical notation (e.g. 2 * 3).
 */
function renderInlineMarkdown(text, keyPrefix = 'inline') {
  if (typeof text !== 'string' || !text) return text;

  // Match inline markdown tokens:
  // 1. ***bold italic***
  // 2. **bold**
  // 3. *italic* (non-whitespace bounded to prevent matching arithmetic multiplication like `2 * 3`)
  // 4. `inline code`
  const tokenRegex = /(\*\*\*(?!\s)[^*]+?(?<!\s)\*\*\*|\*\*(?!\s)[^*]+?(?<!\s)\*\*|\*(?!\s)[^*]+?(?<!\s)\*|`[^`]+`)/g;

  const elements = [];
  let lastIndex = 0;
  let match;

  while ((match = tokenRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      const rawChunk = text.slice(lastIndex, match.index);
      // Clean any accidental dangling unclosed ** markers in raw text chunk
      elements.push(rawChunk.replace(/\*\*/g, ''));
    }

    const token = match[0];
    const key = `${keyPrefix}-${match.index}`;

    if (token.startsWith('***') && token.endsWith('***')) {
      elements.push(
        <strong key={key} style={{ fontWeight: 700, fontStyle: 'italic', color: COLORS.text.primary }}>
          {token.slice(3, -3)}
        </strong>
      );
    } else if (token.startsWith('**') && token.endsWith('**')) {
      elements.push(
        <strong key={key} style={{ fontWeight: 700, color: COLORS.text.primary }}>
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      elements.push(
        <em key={key} style={{ fontStyle: 'italic' }}>
          {token.slice(1, -1)}
        </em>
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      elements.push(
        <code
          key={key}
          style={{
            fontFamily: "'Space Mono', monospace",
            background: 'rgba(255, 255, 255, 0.08)',
            padding: '1px 5px',
            borderRadius: 4,
            fontSize: '0.9em',
          }}
        >
          {token.slice(1, -1)}
        </code>
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    const trailingChunk = text.slice(lastIndex);
    elements.push(trailingChunk.replace(/\*\*/g, ''));
  }

  return elements.length > 0 ? elements : text.replace(/\*\*/g, '');
}

/**
 * Render a markdown-formatted LLM explanation (with ### headings, bullets, and inline emphasis)
 * as styled React elements. Prevents raw ### and ** characters from appearing literally in the UI.
 */
function renderMarkdownExplanation(text) {
  if (!text) return null;
  const lines = text.split('\n');
  const elements = [];
  let currentParagraphLines = [];

  const flushParagraph = (key) => {
    if (currentParagraphLines.length > 0) {
      const joined = currentParagraphLines.join(' ').trim();
      if (joined) {
        elements.push(
          <p key={`p-${key}`} style={{ margin: '0 0 6px 0', lineHeight: 1.65, color: COLORS.text.primary }}>
            {renderInlineMarkdown(joined, `p-${key}`)}
          </p>
        );
      }
      currentParagraphLines = [];
    }
  };

  lines.forEach((line, idx) => {
    const headingMatch = line.match(/^\s*#{1,4}\s+(.+)/);
    const standaloneHeadingMatch = !headingMatch && line.match(/^\s*(?:[^\w\s]+\s*)?\*\*(Situation|Why\s*(?:\/|or)?\s*Possible Causes|Recommended Actions\s*(?:\/|and|&)?\s*Remedies|Precautions|AI Forecast Summary)\*\*:?\s*$/i);
    const bulletMatch = line.match(/^\s*[-*•]\s+(.+)/);
    const numberedMatch = line.match(/^\s*(\d+\.)\s+(.+)/);

    if (headingMatch || standaloneHeadingMatch) {
      flushParagraph(idx);
      const headingRaw = headingMatch ? headingMatch[1] : standaloneHeadingMatch[0];
      const headingText = cleanHeadingText(headingRaw);
      elements.push(
        <div key={`h-${idx}`} style={{
          fontSize: 13,
          fontWeight: 800,
          color: COLORS.primary,
          marginTop: elements.length > 0 ? 14 : 0,
          marginBottom: 6,
          fontFamily: "'Space Grotesk', sans-serif",
          letterSpacing: '0.01em',
        }}>
          {headingText}
        </div>
      );
    } else if (bulletMatch) {
      flushParagraph(idx);
      elements.push(
        <div key={`b-${idx}`} style={{ display: 'flex', gap: 6, marginBottom: 4, color: COLORS.text.primary, fontSize: 14, lineHeight: 1.6 }}>
          <span style={{ color: COLORS.primary, flexShrink: 0 }}>•</span>
          <span>{renderInlineMarkdown(bulletMatch[1], `b-${idx}`)}</span>
        </div>
      );
    } else if (numberedMatch) {
      flushParagraph(idx);
      elements.push(
        <div key={`num-${idx}`} style={{ display: 'flex', gap: 6, marginBottom: 4, color: COLORS.text.primary, fontSize: 14, lineHeight: 1.6 }}>
          <span style={{ color: COLORS.primary, fontWeight: 700, flexShrink: 0, minWidth: 18 }}>{numberedMatch[1]}</span>
          <span>{renderInlineMarkdown(numberedMatch[2], `num-${idx}`)}</span>
        </div>
      );
    } else if (line.trim() === '') {
      flushParagraph(idx);
    } else {
      currentParagraphLines.push(line.trim());
    }
  });
  flushParagraph('end');
  return elements;
}


export default function ForecastDashboard({ initialDomain = 'retail', userRole = 'viewer', accessToken = null }) {
  const [domain, setDomain] = useState(initialDomain);
  const [model, setModel] = useState('auto_select');

  // State Management
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastStage, setForecastStage] = useState('');
  const [forecastData, setForecastData] = useState(null);
  const [forecastError, setForecastError] = useState(null);


  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData, setCompareData] = useState(null);
  const [compareError, setCompareError] = useState(null);

  const [autoSelectedModel, setAutoSelectedModel] = useState(null);
  const [autoSelectReason, setAutoSelectReason] = useState(null);

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

  // Dataset Intelligence Profile State
  const [datasetProfile, setDatasetProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState(null);
  const [selectedTargetOverride, setSelectedTargetOverride] = useState(null);

  // Geographic Scope State
  const [geoMeta, setGeoMeta] = useState(null);
  const [geoFilters, setGeoFilters] = useState({ country: '', state: '', district: '', region: '', city: '', location: '' });
  const [selectedLocation, setSelectedLocation] = useState('');
  const [geoLoading, setGeoLoading] = useState(false);
  const [geoError, setGeoError] = useState(null);

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
      if (nextSelectedId) {
        loadDatasetProfile(nextSelectedId, targetDomain);
        loadDatasetGeography(nextSelectedId, { country: '', state: '', district: '', region: '', city: '', location: '' });
      }
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

  const loadDatasetProfile = async (dsId = selectedDatasetId, targetDomain = domain, targetOverride = null) => {
    if (!dsId) {
      setDatasetProfile(null);
      return;
    }
    setProfileLoading(true);
    setProfileError(null);
    try {
      const res = await profileDataset({
        datasetId: dsId,
        domain: targetDomain,
        target: targetOverride,
        accessToken,
      });
      setDatasetProfile(res);
      if (res?.selected_target && !targetOverride) {
        setSelectedTargetOverride(res.selected_target);
      }
    } catch (err) {
      console.error('Dataset profile error:', err);
      setProfileError(err.message || 'Unable to profile dataset schema.');
      setDatasetProfile(null);
    } finally {
      setProfileLoading(false);
    }
  };

  const loadDatasetGeography = async (dsId = selectedDatasetId, filters = geoFilters, loc = selectedLocation) => {
    if (!dsId) {
      setGeoMeta(null);
      return;
    }
    setGeoLoading(true);
    setGeoError(null);
    try {
      const activeLoc = loc !== undefined ? loc : selectedLocation;
      const res = await fetchDatasetGeography({
        datasetId: dsId,
        geoFilters: filters,
        location: activeLoc || undefined,
        accessToken,
      });
      setGeoMeta(res);
      return res;
    } catch (err) {
      console.error('Dataset geography error:', err);
      setGeoError(err.message || 'Unable to fetch geographic metadata.');
      setGeoMeta(null);
    } finally {
      setGeoLoading(false);
    }
  };


  const getActiveFeaturesDir = (override, activeDsId = selectedDatasetId) => {
    if (override !== undefined) return override;
    const dsObj = datasetsList.find((d) => d.dataset_id === activeDsId);
    if (dsObj && dsObj.features_dir) {
      return dsObj.features_dir;
    }
    return undefined;
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

  // Initial forecast on mount: loads datasets and automatically triggers handleDatasetChange for default dataset
  useEffect(() => {
    const init = async () => {
      const list = await loadDatasets(initialDomain);
      const defaultDs = list[0];
      if (defaultDs?.dataset_id) {
        await handleDatasetChange(defaultDs.dataset_id, initialDomain, 'auto_select');
      }
      loadForecastHistory();
    };
    init();
  }, []);

  // Clear stale state when domain changes
  const handleDomainSwitch = async (newDomain) => {
    if (!['retail', 'electricity'].includes(newDomain)) return;
    domainRequestToken.current += 1;
    setDomain(newDomain);
    setModel('auto_select');
    setUploadDomain(newDomain);
    setGeoFilters({ country: '', state: '', district: '', region: '', city: '', location: '' });
    setSelectedLocation('');
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
    if (defaultDs?.dataset_id) {
      await handleDatasetChange(defaultDs.dataset_id, newDomain, 'auto_select');
    }
  };

  // Unified dynamic dataset change handler
  const handleDatasetChange = async (selectedId, newDomain = undefined, modelOverride = undefined) => {
    const targetDomain = newDomain || domain;
    const targetModel = modelOverride || 'auto_select';
    if (!selectedId) return;
    domainRequestToken.current += 1;
    const requestToken = domainRequestToken.current;

    // Step 1: Wipe all old state and graph immediately
    setSelectedDatasetId(selectedId);
    setModel(targetModel);
    setCustomFeaturesDir(null);
    setGeoMeta(null);
    setForecastData(null);
    setCompareData(null);
    setExplainData(null);
    setFutureData(null);
    setAutoSelectedModel(null);
    setAutoSelectReason(null);
    setForecastError(null);
    setCompareError(null);
    setExplainError(null);
    setFutureError(null);
    setGeoFilters({ country: '', state: '', district: '', region: '', city: '', location: '' });
    setSelectedLocation('');
    setForecastLoading(true);

    // Step 2: Load profile & geography for new dataset
    loadDatasetProfile(selectedId, targetDomain);

    try {
      const geoRes = await fetchDatasetGeography({
        datasetId: selectedId,
        geoFilters: {},
        accessToken,
      });
      if (requestToken !== domainRequestToken.current) return;
      setGeoMeta(geoRes);

      let targetLoc = '';
      let targetFilters = { country: '', state: '', district: '', region: '', city: '', location: '' };

      if (geoRes?.has_geography && geoRes?.available_locations?.length > 0) {
        // Automatically select the FIRST available location from the newly selected dataset
        targetLoc = geoRes.available_locations[0];
        targetFilters = { ...targetFilters, location: targetLoc };
        setSelectedLocation(targetLoc);
        setGeoFilters(targetFilters);
        const locGeo = await loadDatasetGeography(selectedId, targetFilters, targetLoc);
        handleRunForecast(targetModel, targetDomain, undefined, false, targetFilters, targetLoc, locGeo, selectedId);
      } else {
        setSelectedLocation('');
        setGeoFilters(targetFilters);
        handleRunForecast(targetModel, targetDomain, undefined, false, targetFilters, '', geoRes, selectedId);
      }
    } catch (err) {
      console.error('Failed to resolve geography on dataset change:', err);
      if (requestToken !== domainRequestToken.current) return;
      setSelectedLocation('');
      handleRunForecast(targetModel, targetDomain, undefined, false, {}, '', null, selectedId);
    }
  };

  // Location switch runner: immediately clears old location forecast and runs new forecast
  const handleLocationChange = async (nextFilters, locLabel = '') => {
    setGeoFilters(nextFilters);
    setSelectedLocation(locLabel);
    setForecastData(null);
    setCompareData(null);
    setExplainData(null);
    setFutureData(null);
    setForecastError(null);
    setAutoSelectedModel(null);
    setAutoSelectReason(null);
    const freshGeo = await loadDatasetGeography(selectedDatasetId, nextFilters, locLabel);
    handleRunForecast(model, domain, undefined, false, nextFilters, locLabel, freshGeo, selectedDatasetId);
  };


  // Primary forecast runner
  const handleRunForecast = async (
    targetModel = model,
    targetDomain = domain,
    featuresDirOverride = undefined,
    forceRetrain = false,
    overrideGeoFilters = undefined,
    overrideLocation = undefined,
    overrideGeoMeta = undefined,
    overrideDatasetId = undefined,
  ) => {
    const requestToken = domainRequestToken.current;
    const activeDatasetId = overrideDatasetId !== undefined ? overrideDatasetId : selectedDatasetId;
    const activeGeo = overrideGeoFilters !== undefined ? overrideGeoFilters : geoFilters;
    const activeLoc = overrideLocation !== undefined ? overrideLocation : selectedLocation;
    const activeGeoMeta = overrideGeoMeta !== undefined ? overrideGeoMeta : geoMeta;

    setForecastLoading(true);
    setForecastError(null);
    setExplainData(null);
    setExplainError(null);
    setForecastData(null); // Clear previous graph immediately
    setAutoSelectedModel(null);
    setAutoSelectReason(null);

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
            datasetId: activeDatasetId,
            target: selectedTargetOverride,
            location: activeLoc || undefined,
            geoFilters: activeGeo,
            featuresDir: activeFeaturesDir,
          });
          if (requestToken !== domainRequestToken.current) return;
          setCompareData(compResult);
          actualModel = compResult.recommended_model;
          if (!actualModel) {
            throw new Error('No eligible models available for comparison on this dataset.');
          }
          recModel = actualModel;
          setAutoSelectedModel(recModel);
        } catch (err) {
          console.error('Auto-select comparison failed:', err);
          if (requestToken !== domainRequestToken.current) return;
          if (isApplicabilityNaError(err.message)) {
            const details = parseApplicabilityDetails(err.message, err.data || activeGeoMeta);
            setAutoSelectedModel('n/a');
            setAutoSelectReason(details);
            setForecastError(null);
            return;
          }
          setAutoSelectedModel(null);
          setAutoSelectReason(null);
          setForecastError(err.message || 'Auto Select comparison failed. Please verify backend service.');
          return;
        } finally {
          setCompareLoading(false);
        }
      } else {
        setAutoSelectedModel(null);
        setAutoSelectReason(null);
      }

      // Execute forecast call
      const res = await runForecast({
        domain: targetDomain,
        model: actualModel,
        datasetId: activeDatasetId,
        target: selectedTargetOverride,
        location: activeLoc || undefined,
        geoFilters: activeGeo,
        featuresDir: activeFeaturesDir,
        predictionLimit: 100,
        selectionMethod: targetModel === 'auto_select' ? 'validation_rmse' : undefined,
        forceRetrain,
      });

      if (requestToken !== domainRequestToken.current) return;
      setForecastData(res);
      loadForecastHistory();

      // Trigger SHAP explainability for all supported tree/hybrid models
      if (['random_forest', 'xgboost', 'trend_xgboost', 'hybrid'].includes(actualModel)) {
        if (targetDomain === 'electricity') {
          // For Electricity, display available model-native feature importances immediately
          // without waiting for the expensive detailed SHAP TreeExplainer calculation.
          // Detailed SHAP remains available on-demand via the explicit button below.
          if (res?.feature_importances && Object.keys(res.feature_importances).length > 0) {
            setExplainData({
              model: actualModel,
              domain: targetDomain,
              method: 'native',
              feature_importance: res.feature_importances,
              dataset_context: res.dataset_context,
              natural_language_explanation: 'Model-native feature importances (Gini/gain importance) computed during training. Detailed SHAP TreeExplainer values can be computed on demand below.',
              is_native: true,
            });
          }
        } else {
          // Retail and other domains retain existing automatic SHAP calculation unchanged
          fetchExplainability(targetDomain, actualModel, activeFeaturesDir, activeGeo, activeLoc, activeDatasetId);
        }
      }
    } catch (err) {
      console.error('Forecast error:', err);
      if (isApplicabilityNaError(err.message)) {
        setForecastError(formatUserFacingError(err.message, err.data || activeGeoMeta, activeLoc));
      } else {
        setForecastError(err.message || 'Forecast request failed. Please verify backend service.');
      }
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
        datasetId: selectedDatasetId,
        target: selectedTargetOverride,
        location: selectedLocation || undefined,
        geoFilters,
        featuresDir: getActiveFeaturesDir(),
      });
      if (requestToken === domainRequestToken.current) setCompareData(res);
    } catch (err) {
      console.error('Comparison error:', err);
      if (isApplicabilityNaError(err.message)) {
        setCompareError(formatUserFacingError(err.message, err.data || geoMeta, selectedLocation));
      } else {
        setCompareError(err.message || 'Model comparison request failed. Please verify backend service.');
      }
    } finally {
      setCompareLoading(false);
    }
  };

  // Explainability runner
  const fetchExplainability = async (
    expDomain,
    expModel,
    featuresDirOverride = undefined,
    expGeo = geoFilters,
    expLoc = selectedLocation,
    expDatasetId = undefined,
  ) => {
    const requestToken = domainRequestToken.current;
    setExplainLoading(true);
    setExplainError(null);
    try {
      const res = await explainForecast({
        domain: expDomain,
        model: expModel,
        datasetId: expDatasetId || selectedDatasetId,
        target: selectedTargetOverride,
        location: expLoc || undefined,
        geoFilters: expGeo,
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
        datasetId: selectedDatasetId,
        target: selectedTargetOverride,
        location: selectedLocation || undefined,
        geoFilters,
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
      await loadDatasets(uploadDomain, uploaded.dataset_id);
      await handleDatasetChange(uploaded.dataset_id, uploadDomain, 'auto_select');
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
    ? (autoSelectedModel === 'n/a' ? 'Auto Select (N/A — No Applicable Model)' : `${autoSelectedModel.replace('_', ' ')} (Auto Selected)`)
    : (forecastData?.model || model).replace('_', ' ');

  const isExplainableModel = ['random_forest', 'xgboost', 'trend_xgboost', 'lstm', 'cnn_lstm', 'hybrid'].includes(forecastData?.model || model);

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
              onChange={(e) => handleDatasetChange(e.target.value)}
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
                datasetsList.map((d) => {
                  // Build a meaningful identity suffix for uploaded datasets.
                  // Prefer content_hash prefix (6 hex chars) so identical files
                  // resolve to the same label; fall back to dataset_id prefix.
                  let suffix = d.is_demo ? '(Demo)' : '';
                  if (!d.is_demo && d.dataset_id) {
                    const hashPrefix = d.content_hash
                      ? `#${d.content_hash.slice(0, 6)}`
                      : d.dataset_id.slice(0, 8);
                    suffix = `(Uploaded ${hashPrefix})`;
                  }
                  return (
                    <option key={d.dataset_id} value={d.dataset_id}>
                      {d.filename || d.dataset_id} {suffix}
                    </option>
                  );
                })
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
              <option value="auto_select">⚡ Auto Select / Best Model (Validation Recommended)</option>
              <option value="naive">Naive (Reference Baseline)</option>
              <option value="seasonal_naive">Seasonal Naive (Periodic Baseline)</option>
              <option value="arima">ARIMA (Autoregressive Integrated Moving Average)</option>
              <option value="sarima">SARIMA (Seasonal ARIMA)</option>
              <option value="random_forest">Random Forest (Tree Ensemble)</option>
              <option value="xgboost">XGBoost (Gradient Boosted Trees)</option>
              <option value="trend_xgboost">Trend XGBoost (Piecewise Trend + Detrending)</option>
              <option value="lstm">LSTM (PyTorch Deep Sequence)</option>
              <option value="cnn_lstm">CNN-LSTM (Deep Hybrid Sequence)</option>
              <option value="hybrid">Hybrid Ensemble (Validation-Weighted)</option>
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

        {/* Dataset Intelligence Profile Card */}
        {selectedDatasetId && (
          <div
            style={{
              background: COLORS.bg.dark,
              border: `1px solid ${COLORS.bg.border}`,
              borderRadius: 12,
              padding: 16,
              marginTop: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>🧠</span>
                <span style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary }}>Dataset Intelligence</span>
                <span style={{ fontSize: 12, color: COLORS.text.muted }}>Automated schema inspection & suitability score</span>
              </div>
              {datasetProfile && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 12, color: COLORS.text.secondary }}>Domain Fit:</span>
                  <span
                    style={{
                      background:
                        datasetProfile.confidence_label === 'High'
                          ? `${COLORS.success}22`
                          : datasetProfile.confidence_label === 'Medium'
                          ? `${COLORS.warning}22`
                          : `${COLORS.accent}22`,
                      color:
                        datasetProfile.confidence_label === 'High'
                          ? COLORS.success
                          : datasetProfile.confidence_label === 'Medium'
                          ? COLORS.warning
                          : COLORS.accent,
                      border: `1px solid ${
                        datasetProfile.confidence_label === 'High'
                          ? COLORS.success
                          : datasetProfile.confidence_label === 'Medium'
                          ? COLORS.warning
                          : COLORS.accent
                      }44`,
                      borderRadius: 6,
                      padding: '2px 8px',
                      fontSize: 11,
                      fontWeight: 700,
                      textTransform: 'uppercase',
                    }}
                  >
                    {domain} — {datasetProfile.confidence_label} ({Math.round(datasetProfile.confidence * 100)}%)
                  </span>
                </div>
              )}
            </div>

            {profileLoading ? (
              <div style={{ fontSize: 13, color: COLORS.text.muted, padding: '8px 0' }}>
                Analyzing dataset schema and profiling features...
              </div>
            ) : profileError ? (
              <div style={{ fontSize: 13, color: COLORS.accent, padding: '6px 0' }}>{profileError}</div>
            ) : datasetProfile ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {/* Compact Data Quality & Readiness Summary Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Rows & Columns</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginTop: 2 }}>
                      {formatNumber(datasetProfile.row_count, 0)} rows × {datasetProfile.column_count} cols
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Date Column</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.primary, marginTop: 2 }}>
                      📅 {datasetProfile.time_column || 'None Detected'}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Target Column</div>
                    {datasetProfile.target_candidates && datasetProfile.target_candidates.length > 1 ? (
                      <select
                        value={selectedTargetOverride || datasetProfile.selected_target || ''}
                        onChange={(e) => {
                          const target = e.target.value;
                          setSelectedTargetOverride(target);
                          loadDatasetProfile(selectedDatasetId, domain, target);
                        }}
                        style={{
                          width: '100%',
                          background: COLORS.bg.dark,
                          border: `1px solid ${COLORS.primary}`,
                          color: COLORS.success,
                          fontWeight: 700,
                          fontSize: 12,
                          borderRadius: 6,
                          padding: '2px 4px',
                          marginTop: 2,
                          outline: 'none',
                        }}
                      >
                        {datasetProfile.target_candidates.map((t) => (
                          <option key={t} value={t}>
                            🎯 {t}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.success, marginTop: 2 }}>
                        🎯 {datasetProfile.selected_target || 'None Detected'}
                      </div>
                    )}
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Unique Timestamps</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginTop: 2 }}>
                      ⏱️ {formatNumber(datasetProfile.unique_timestamps || datasetProfile.suitability?.unique_timestamps || 0, 0)}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Duplicates Removed</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginTop: 2 }}>
                      🧹 {formatNumber(datasetProfile.duplicate_rows_removed || datasetProfile.duplicate_rows || 0, 0)}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Timestamps Handled</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginTop: 2 }}>
                      🔄 {formatNumber(datasetProfile.duplicate_timestamps_count || 0, 0)}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Missing Imputed</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginTop: 2 }}>
                      ✨ {formatNumber(datasetProfile.missing_values_handled || 0, 0)}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Excluded Features</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.warning, marginTop: 2 }}>
                      🛡️ {datasetProfile.excluded_features_count ?? datasetProfile.ignored_columns?.length ?? 0}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Forecast Features</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.primary, marginTop: 2 }}>
                      📊 {datasetProfile.final_forecasting_features_count ?? 0}
                    </div>
                  </div>

                  <div style={{ background: COLORS.bg.card, padding: 10, borderRadius: 8, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: 'uppercase' }}>Suitability Status</div>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color:
                          (datasetProfile.suitability?.classification || (datasetProfile.suitable ? 'SUITABLE' : 'UNSUITABLE')) === 'SUITABLE'
                            ? COLORS.success
                            : (datasetProfile.suitability?.classification === 'SUITABLE_WITH_WARNINGS' ? COLORS.warning : COLORS.accent),
                        marginTop: 2,
                      }}
                    >
                      {datasetProfile.suitability?.status_label || (datasetProfile.suitable ? '✓ Fully Suitable' : '⚠️ Unsuitable')}
                    </div>
                  </div>
                </div>

                {/* Explanation & Warnings/Reasons */}
                <div style={{ background: COLORS.bg.card, borderRadius: 8, padding: 12, border: `1px solid ${COLORS.bg.border}`, fontSize: 12, color: COLORS.text.secondary, lineHeight: 1.5 }}>
                  <div style={{ fontWeight: 600, color: COLORS.text.primary, marginBottom: 4 }}>
                    {datasetProfile.explanation}
                  </div>
                  {datasetProfile.suitability?.warnings?.length > 0 && (
                    <div style={{ color: COLORS.warning, marginTop: 4 }}>
                      ⚠️ <strong>Warnings:</strong> {datasetProfile.suitability.warnings.join(' | ')}
                    </div>
                  )}
                  {datasetProfile.suitability?.reasons?.length > 0 && (
                    <div style={{ color: COLORS.accent, marginTop: 4 }}>
                      ⛔ <strong>Unsuitability Reasons:</strong> {datasetProfile.suitability.reasons.join(' | ')}
                    </div>
                  )}
                </div>
              </div>
            ) : null}


          </div>
        )}

        {/* Geographic Scope Card */}
        {selectedDatasetId && geoMeta && (
          <div
            style={{
              background: COLORS.bg.dark,
              border: `1px solid ${COLORS.bg.border}`,
              borderRadius: 12,
              padding: 16,
              marginTop: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, flexWrap: 'wrap', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>🌍</span>
                <span style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary }}>Geographic Scope</span>
                <span style={{ fontSize: 12, color: COLORS.text.muted }}>Cascading location filters</span>
              </div>
              {geoMeta.has_geography ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  {selectedLocation && (
                    <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.success, background: `${COLORS.success}18`, padding: '2px 8px', borderRadius: 6, border: `1px solid ${COLORS.success}44` }}>
                      📍 Active: {selectedLocation}
                    </span>
                  )}
                  <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.primary, background: `${COLORS.primary}18`, padding: '2px 8px', borderRadius: 6, border: `1px solid ${COLORS.primary}44` }}>
                    📊 {formatNumber(geoMeta.filtered_rows, 0)} records {selectedLocation ? `(${geoMeta.training_rows ?? (geoMeta.filtered_rows ? Math.floor(geoMeta.filtered_rows * 0.70) : 0)} training obs.)` : ''} / {formatNumber(geoMeta.total_rows, 0)} total
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: 12, color: COLORS.text.secondary }}>
                  ℹ️ Geographic filtering is not applicable for this dataset.
                </div>
              )}
            </div>

            {geoMeta.has_geography ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 10 }}>
                {/* Direct / Quick Location Selector */}
                {geoMeta.available_locations?.length > 0 && (
                  <div style={{ background: COLORS.bg.card, borderRadius: 10, padding: 12, border: `1px solid ${COLORS.bg.border}` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexWrap: 'wrap', gap: 6 }}>
                      <label style={{ fontSize: 11, fontWeight: 800, color: COLORS.primary, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                        🎯 SELECT SPECIFIC LOCATION / PLACE
                      </label>
                      {selectedLocation && (
                        <button
                          type="button"
                          onClick={() => handleLocationChange({ country: '', state: '', district: '', region: '', city: '', location: '' }, '')}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: COLORS.accent,
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: 'pointer',
                            textDecoration: 'underline',
                          }}
                        >
                          Reset to All Geographies
                        </button>
                      )}
                    </div>
                    <select
                      value={selectedLocation || ''}
                      onChange={(e) => {
                        const loc = e.target.value;
                        if (!loc) {
                          handleLocationChange({ country: '', state: '', district: '', region: '', city: '', location: '' }, '');
                        } else {
                          handleLocationChange({ ...geoFilters, location: loc }, loc);
                        }
                      }}
                      style={{
                        width: '100%',
                        background: COLORS.bg.dark,
                        border: `1px solid ${selectedLocation ? COLORS.primary : COLORS.bg.border}`,
                        borderRadius: 8,
                        padding: '8px 12px',
                        color: COLORS.text.primary,
                        fontSize: 13,
                        fontWeight: 700,
                      }}
                    >
                      <option value="">-- All Geographies (Global Dataset) --</option>
                      {geoMeta.available_locations.map((loc) => (
                        <option key={loc} value={loc}>📍 {loc}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Dynamic Hierarchy Filters derived from dataset schema */}
                {geoMeta.hierarchy_levels && geoMeta.hierarchy_levels.length > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                    {geoMeta.hierarchy_levels.map((lvl) => (
                      <div key={lvl.level_key}>
                        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 4, textTransform: 'uppercase' }}>
                          {lvl.label}
                        </label>
                        <select
                          value={geoFilters[lvl.level_key] || ''}
                          onChange={(e) => {
                            const val = e.target.value;
                            const next = { ...geoFilters, [lvl.level_key]: val };
                            handleLocationChange(next, val || '');
                          }}
                          style={{
                            width: '100%',
                            background: COLORS.bg.card,
                            border: `1px solid ${COLORS.bg.border}`,
                            borderRadius: 8,
                            padding: '6px 10px',
                            color: COLORS.text.primary,
                            fontSize: 12,
                          }}
                        >
                          <option value="">All {lvl.label}s</option>
                          {lvl.values?.map((v) => (
                            <option key={v} value={v}>{v}</option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                    {geoMeta.levels?.countries?.length > 0 && (
                      <div>
                        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 4 }}>COUNTRY</label>
                        <select
                          value={geoFilters.country || ''}
                          onChange={(e) => {
                            const val = e.target.value;
                            const next = { country: val, state: '', district: '', region: '', city: '', location: '' };
                            handleLocationChange(next, val);
                          }}
                          style={{ width: '100%', background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 8, padding: '6px 10px', color: COLORS.text.primary, fontSize: 12 }}
                        >
                          <option value="">All Countries</option>
                          {geoMeta.levels.countries.map((c) => (<option key={c} value={c}>{c}</option>))}
                        </select>
                      </div>
                    )}
                    {geoMeta.levels?.states?.length > 0 && (
                      <div>
                        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 4 }}>STATE</label>
                        <select
                          value={geoFilters.state || ''}
                          onChange={(e) => {
                            const val = e.target.value;
                            const next = { ...geoFilters, state: val, district: '', region: '', city: '', location: '' };
                            handleLocationChange(next, val || geoFilters.country || '');
                          }}
                          style={{ width: '100%', background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 8, padding: '6px 10px', color: COLORS.text.primary, fontSize: 12 }}
                        >
                          <option value="">All States</option>
                          {geoMeta.levels.states.map((s) => (<option key={s} value={s}>{s}</option>))}
                        </select>
                      </div>
                    )}
                    {geoMeta.levels?.districts?.length > 0 && (
                      <div>
                        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 4 }}>DISTRICT</label>
                        <select
                          value={geoFilters.district || ''}
                          onChange={(e) => {
                            const val = e.target.value;
                            const next = { ...geoFilters, district: val, city: '', location: '' };
                            handleLocationChange(next, val || geoFilters.state || '');
                          }}
                          style={{ width: '100%', background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 8, padding: '6px 10px', color: COLORS.text.primary, fontSize: 12 }}
                        >
                          <option value="">All Districts</option>
                          {geoMeta.levels.districts.map((d) => (<option key={d} value={d}>{d}</option>))}
                        </select>
                      </div>
                    )}
                    {geoMeta.levels?.regions?.length > 0 && (
                      <div>
                        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 4 }}>REGION</label>
                        <select
                          value={geoFilters.region || ''}
                          onChange={(e) => {
                            const val = e.target.value;
                            const next = { ...geoFilters, region: val, city: '', location: '' };
                            handleLocationChange(next, val || geoFilters.state || '');
                          }}
                          style={{ width: '100%', background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 8, padding: '6px 10px', color: COLORS.text.primary, fontSize: 12 }}
                        >
                          <option value="">All Regions</option>
                          {geoMeta.levels.regions.map((r) => (<option key={r} value={r}>{r}</option>))}
                        </select>
                      </div>
                    )}
                    {geoMeta.levels?.cities?.length > 0 && (
                      <div>
                        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, marginBottom: 4 }}>CITY / AREA</label>
                        <select
                          value={geoFilters.city || ''}
                          onChange={(e) => {
                            const val = e.target.value;
                            const next = { ...geoFilters, city: val, location: '' };
                            handleLocationChange(next, val || geoFilters.district || geoFilters.state || '');
                          }}
                          style={{ width: '100%', background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 8, padding: '6px 10px', color: COLORS.text.primary, fontSize: 12 }}
                        >
                          <option value="">All Cities</option>
                          {geoMeta.levels.cities.map((ci) => (<option key={ci} value={ci}>{ci}</option>))}
                        </select>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: COLORS.text.secondary, background: COLORS.bg.card, padding: 12, borderRadius: 8, border: `1px solid ${COLORS.bg.border}`, marginTop: 8 }}>
                ℹ️ <strong>Dataset contains no geographic columns.</strong> The forecasting pipeline, model selection, SHAP attribution, and charts are computed across the full dataset without geographic subdivision.
              </div>
            )}
          </div>
        )}



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
        {autoSelectedModel && autoSelectedModel !== 'n/a' && (
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

        {autoSelectedModel === 'n/a' && (
          <div style={{ background: `${COLORS.warning}15`, border: `1px solid ${COLORS.warning}44`, borderRadius: 10, padding: '14px 18px', display: 'flex', alignItems: 'flex-start', gap: 14 }}>
            <span style={{ fontSize: 24, lineHeight: 1 }}>⚠️</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.warning }}>
                Auto Select: N/A — No Applicable Forecasting Model
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.text.primary, marginTop: 4, lineHeight: 1.4 }}>
                {autoSelectReason?.criterion ||
                  `${autoSelectReason?.training_observations ?? geoMeta?.training_observations ?? geoMeta?.training_rows ?? (geoMeta?.filtered_rows ? Math.floor(geoMeta.filtered_rows * 0.70) : 0)} usable training observations available; minimum model requirements are not met.`}
              </div>
              <div style={{ fontSize: 12, color: COLORS.text.secondary || '#94a3b8', marginTop: 4, lineHeight: 1.4 }}>
                {autoSelectReason?.explanation ||
                  'Forecasting cannot be performed for the selected location because the available training history does not satisfy the minimum requirements of the applicable models.'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Presentation-Safe Error Alert */}
      {forecastError && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: '16px 20px', color: COLORS.accent, fontSize: 14, fontWeight: 600 }}>
          ⚠️ {formatUserFacingError(forecastError, geoMeta, selectedLocation)}
        </div>
      )}

      {/* ─── Forecast Summary Context Banner ─── */}
      {forecastData?.dataset_context && (
        <div
          style={{
            background: COLORS.bg.card,
            border: `1px solid ${COLORS.primary}44`,
            borderRadius: 12,
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 20 }}>📋</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary }}>
                Forecast Summary Context: {forecastData.dataset_context.dataset_name}
              </div>
              <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 2 }}>
                Dataset ID: <strong style={{ color: COLORS.primary, fontFamily: "'Space Mono', monospace" }}>{forecastData.dataset_context.dataset_id}</strong> | Hash: <strong style={{ color: COLORS.secondary, fontFamily: "'Space Mono', monospace" }}>{forecastData.dataset_context.dataset_hash}</strong> | Target: <strong style={{ color: COLORS.primary }}>{forecastData.dataset_context.target}</strong> | Scope: <strong style={{ color: COLORS.success }}>{forecastData.dataset_context.geography_label}</strong> {forecastData.dataset_context.rows_used > 0 ? `(${formatNumber(forecastData.dataset_context.rows_used, 0)} rows used)` : ''}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 12, color: COLORS.text.secondary }}>Model: <strong style={{ color: COLORS.text.primary, textTransform: 'capitalize' }}>{forecastData.dataset_context.model}</strong></span>
            <span
              style={{
                background: forecastData.dataset_context.artifact_reused ? `${COLORS.success}22` : `${COLORS.warning}22`,
                color: forecastData.dataset_context.artifact_reused ? COLORS.success : COLORS.warning,
                border: `1px solid ${forecastData.dataset_context.artifact_reused ? COLORS.success : COLORS.warning}44`,
                padding: '2px 8px',
                borderRadius: 6,
                fontSize: 11,
                fontWeight: 700,
              }}
            >
              {forecastData.dataset_context.artifact_reused ? '⚡ Artifact Reused' : '🏋️ Model Trained'}
            </span>
          </div>
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
      {/* ─── 3. Metric Cards & Baseline Comparison ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14 }}>
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
            sub: forecastData?.baseline_metrics ? `Baseline: ${formatNumber(forecastData.baseline_metrics.rmse)} (${forecastData.improvement_pct ?? 0}% imp)` : null,
            desc: 'Root mean squared error penalizing larger forecasting deviations.',
            color: COLORS.secondary,
          },
          {
            label: 'MAPE',
            title: 'Mean Abs % Error',
            value: formatMape(forecastData?.test_metrics?.mape),
            desc: 'Average percentage forecasting error across test horizon.',
            color: COLORS.warning,
          },
          {
            label: 'SMAPE / WAPE',
            title: 'Symmetric & Weighted MAPE',
            value: `${formatMape(forecastData?.test_metrics?.smape)} / ${formatMape(forecastData?.test_metrics?.wape)}`,
            desc: 'Symmetric (SMAPE) and Weighted (WAPE) percentage errors (stable near zero).',
            color: COLORS.success,
          },
          {
            label: 'R² Score',
            title: 'Coefficient of Determination',
            value: formatNumber(forecastData?.test_metrics?.r2, 4),
            desc: 'Explained variance / goodness of fit (1.0 = perfect fit).',
            color: (forecastData?.test_metrics?.r2 ?? 0) < 0 ? COLORS.accent : COLORS.warning,
          },
          {
            label: 'Forecast Quality',
            title: 'Overall Model Readiness',
            value: forecastData?.forecast_quality || ((forecastData?.test_metrics?.r2 ?? 0) < 0 ? 'Needs Improvement' : 'Good'),
            desc: 'Validation-checked model readiness status.',
            color: (forecastData?.forecast_quality === 'Needs Improvement' || (forecastData?.test_metrics?.r2 ?? 0) < 0) ? COLORS.accent : COLORS.success,
          },
        ].map((m, idx) => (
          <div
            key={idx}
            style={{
              background: COLORS.bg.card,
              border: `1px solid ${COLORS.bg.border}`,
              borderRadius: 14,
              padding: '16px',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  {m.label}
                </span>
                <span style={{ fontSize: 11, color: COLORS.text.muted }} title={m.desc}>ℹ️</span>
              </div>
              <div style={{ fontSize: 20, fontWeight: 800, color: m.color, fontFamily: "'Space Mono', monospace", margin: '6px 0' }}>
                {forecastLoading ? '...' : m.value}
              </div>
              {m.sub && (
                <div style={{ fontSize: 10, color: COLORS.text.muted, marginTop: 2 }}>
                  {m.sub}
                </div>
              )}
            </div>
            <div style={{ fontSize: 10, color: COLORS.text.muted, marginTop: 6, lineHeight: 1.3 }}>
              {m.desc}
            </div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11, color: COLORS.text.muted, marginTop: -6, marginBottom: 8, fontStyle: 'italic' }}>
        * Note: MAPE can become unstable when actual demand values are zero or very small. SMAPE and WAPE are therefore also reported.
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

      {/* ─── AI Explanation / Explainable AI Section ─── */}
      {forecastData?.llm_explanation && (
        <div
          style={{
            background: COLORS.bg.card,
            border: `1px solid ${COLORS.primary}66`,
            borderRadius: 16,
            padding: 24,
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
            boxShadow: `0 8px 32px ${COLORS.primary}15`,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 22 }}>🤖</span>
              <h3 style={{ fontSize: 18, fontWeight: 800, color: COLORS.text.primary, margin: 0, fontFamily: "'Space Grotesk', sans-serif" }}>
                AI Explanation
              </h3>
            </div>
          </div>

          {/* Natural Language Explanation Text — rendered as structured sections */}
          <div
            style={{
              background: COLORS.bg.dark,
              border: `1px solid ${COLORS.bg.border}`,
              borderRadius: 12,
              padding: 18,
              fontSize: 14,
              color: COLORS.text.primary,
              lineHeight: 1.6,
            }}
          >
            {renderMarkdownExplanation(forecastData.llm_explanation.explanation)}
          </div>


          {/* Structured Summary Cards */}
          {forecastData.llm_explanation.structured_summary && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginTop: 4 }}>
              {/* Selected Forecast Model */}
              <div style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 10, padding: 14 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.primary, textTransform: 'uppercase', marginBottom: 4 }}>
                  Selected Forecast Model
                </div>
                <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary }}>
                  {forecastData.llm_explanation.structured_summary.selected_model?.replace('_', ' ').toUpperCase()}
                </div>
                <div style={{ fontSize: 12, color: COLORS.text.secondary, marginTop: 4 }}>
                  Basis: {forecastData.llm_explanation.structured_summary.selection_reason}
                </div>
              </div>

              {/* Performance Summary */}
              <div style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 10, padding: 14 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.success, textTransform: 'uppercase', marginBottom: 4 }}>
                  Performance Summary
                </div>
                <div style={{ fontSize: 12, color: COLORS.text.secondary, lineHeight: 1.5 }}>
                  R²: <strong style={{ color: COLORS.text.primary }}>{forecastData.llm_explanation.structured_summary.performance_summary?.r2_description}</strong>
                  <br />
                  Baseline Imp: <strong style={{ color: COLORS.success }}>{forecastData.llm_explanation.structured_summary.performance_summary?.baseline_improvement_pct != null ? `${forecastData.llm_explanation.structured_summary.performance_summary.baseline_improvement_pct}%` : 'N/A'}</strong>
                </div>
              </div>

              {/* Important Drivers */}
              <div style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 10, padding: 14 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.warning, textTransform: 'uppercase', marginBottom: 4 }}>
                  Important Signals (SHAP)
                </div>
                <div style={{ fontSize: 12, color: COLORS.text.secondary, lineHeight: 1.5 }}>
                  {forecastData.llm_explanation.structured_summary.important_drivers?.length > 0
                    ? forecastData.llm_explanation.structured_summary.important_drivers.map(formatFeatureName).join(', ')
                    : (['random_forest', 'xgboost'].includes((forecastData?.model || '').toLowerCase())
                        ? 'Historical demand signals & lag features'
                        : 'SHAP feature attribution is available for tree-based models (Random Forest, XGBoost)')}
                </div>
              </div>

              {/* Data Quality & Reliability */}
              <div style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 10, padding: 14 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.secondary, textTransform: 'uppercase', marginBottom: 4 }}>
                  Data Quality & Reliability
                </div>
                <div style={{ fontSize: 12, color: COLORS.text.secondary, lineHeight: 1.4 }}>
                  {forecastData.llm_explanation.structured_summary.data_quality_warnings?.slice(0, 3).map((w, idx) => (
                    <div key={idx}>✓ {w}</div>
                  ))}
                </div>
              </div>
            </div>
          )}
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
            <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>Actual vs Predicted Demand Series (Held-Out Test Set)</span>
              {selectedLocation && (
                <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.success, background: `${COLORS.success}20`, padding: '2px 8px', borderRadius: 6, border: `1px solid ${COLORS.success}44` }}>
                  📍 {selectedLocation}
                </span>
              )}
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
              <option value="random_forest">Random Forest (Tree Ensemble)</option>
              <option value="xgboost">XGBoost (Gradient Boosted Trees)</option>
              <option value="trend_xgboost">Trend XGBoost (Piecewise Trend)</option>
              <option value="arima">ARIMA (Time Series)</option>
              <option value="sarima">SARIMA (Seasonal Time Series)</option>
              <option value="hybrid">Hybrid Ensemble (Validation-Weighted)</option>
              <option value="lstm">LSTM (PyTorch Deep Sequence)</option>
              <option value="cnn_lstm">CNN-LSTM (Deep Hybrid Sequence)</option>
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
            Future forecasting is not currently supported for model {futureModel}. Choose Random Forest, XGBoost, or Hybrid{domain === 'electricity' ? ' (or ARIMA / SARIMA / Trend XGBoost)' : ''}.
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
              Top Forecast Drivers ({explainData?.is_native ? 'Model Feature Importance' : 'Explainable AI — SHAP'})
            </h3>
            <p style={{ fontSize: 12, color: COLORS.text.secondary, margin: '2px 0 0' }}>
              {explainData?.is_native
                ? 'Relative feature importance score (higher values indicate greater influence on predictions).'
                : 'Mean |SHAP Value| — Higher values indicate features with greater influence on predictions.'}
            </p>
          </div>
          {isExplainableModel && (
            <button
              onClick={() => fetchExplainability(
                domain,
                forecastData?.model || model,
                getActiveFeaturesDir(),
                geoFilters,
                selectedLocation,
                selectedDatasetId,
              )}
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
              {explainLoading ? 'Computing SHAP...' : (explainData?.is_native ? '⚡ Compute Detailed SHAP' : 'Recompute SHAP')}
            </button>
          )}
        </div>

        {!isExplainableModel ? (
          <div style={{ padding: '20px', background: COLORS.bg.dark, borderRadius: 12, border: `1px solid ${COLORS.bg.border}`, color: COLORS.text.secondary, fontSize: 13 }}>
            ℹ️ Feature explainability via Tree SHAP is available for <span style={{ color: COLORS.primary, fontWeight: 700 }}>Random Forest</span>, <span style={{ color: COLORS.primary, fontWeight: 700 }}>XGBoost</span>, and <span style={{ color: COLORS.primary, fontWeight: 700 }}>Hybrid</span> ensemble models. Switch to one of these models to inspect top feature drivers.
          </div>
        ) : explainLoading ? (
          <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: COLORS.text.secondary }}>
            Computing SHAP values across evaluation dataset...
          </div>
        ) : explainError ? (
          <div style={{ color: COLORS.accent, fontSize: 13 }}>⚠️ {explainError}</div>
        ) : explainData ? (() => {
          const rawImp = explainData.feature_importance || explainData.global_feature_importance || explainData.local_attributions;
          const listImp = explainData.features || explainData.feature_importances;

          const featureImportances = rawImp
            ? Object.entries(rawImp).map(([feature, importance]) => ({
                feature: formatFeatureName(feature),
                rawFeature: feature,
                importance: Math.abs(Number(importance)),
              })).sort((a, b) => b.importance - a.importance)
            : (listImp || []).map((item) => ({
                feature: formatFeatureName(item.feature),
                rawFeature: item.feature,
                importance: Math.abs(Number(item.importance)),
              })).sort((a, b) => b.importance - a.importance);

          if (featureImportances.length === 0) {
            return (
              <div style={{ padding: '20px 0', color: COLORS.text.muted, fontSize: 13 }}>
                No feature importances returned for model {explainData.model}.
              </div>
            );
          }

          return (
            <div>
              <div style={{ fontSize: 13, color: COLORS.text.secondary, marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <span>
                  Top feature drivers for model <span style={{ color: COLORS.primary, fontWeight: 700, textTransform: 'capitalize' }}>{explainData.model?.replace('_', ' ')}</span>:
                </span>
                {explainData.dataset_context && (
                  <span style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: COLORS.success, background: `${COLORS.success}18`, padding: '2px 8px', borderRadius: 6, border: `1px solid ${COLORS.success}44` }}>
                    🔒 Verified Dataset: {explainData.dataset_context.dataset_id} (#{explainData.dataset_context.dataset_hash})
                  </span>
                )}
              </div>

              {explainData.natural_language_explanation && (
                <div style={{ background: COLORS.bg.dark, border: `1px solid ${COLORS.primary}44`, borderRadius: 8, padding: '10px 14px', fontSize: 12, color: COLORS.text.primary, marginBottom: 16, lineHeight: 1.5 }}>
                  💬 <strong style={{ color: COLORS.primary }}>Natural Language Attribution:</strong> {explainData.natural_language_explanation}
                </div>
              )}

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
                  <Tooltip contentStyle={customTooltipStyle} formatter={(val) => [formatNumber(val, 4), explainData?.is_native ? 'Importance' : 'Mean |SHAP|']} />
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
            Click "{isExplainableModel && domain === 'electricity' ? '⚡ Compute Detailed SHAP' : 'Recompute SHAP'}" to view feature importance analysis.
          </div>
        )}
      </div>
    </div>
  );
}
