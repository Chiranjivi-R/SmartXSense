/**
 * API client service for SmartXSense ML Forecasting and Explainability endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export const DOMAIN_FEATURE_ARTIFACTS = {
  retail: 'ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4',
  electricity: 'ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25',
};

export const FUTURE_HORIZON_PRESETS = {
  retail: [7, 14, 30],
  electricity: [24, 168, 720, 2160],
};

export const FUTURE_SUPPORTED_MODELS = {
  retail: ['random_forest', 'xgboost', 'hybrid'],
  electricity: ['random_forest', 'xgboost', 'trend_xgboost', 'arima', 'sarima', 'hybrid'],
};

async function parseJsonResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      response.ok
        ? 'Received an invalid response from the SmartXSense backend.'
        : `Request failed with status ${response.status}`,
    );
  }
  if (!response.ok || data.status === 'error') {
    const details = data.errors ? ` ${JSON.stringify(data.errors)}` : '';
    const baseMessage = data.message || `Request failed with status ${response.status}`;
    // Include HTTP status code so the UI can display: "HTTP 500 — ..."
    const statusPrefix = !response.ok ? `HTTP ${response.status} — ` : '';
    const err = new Error(`${statusPrefix}${baseMessage}${details}`);
    err.status = response.status;
    err.data = data.data;
    err.raw = data;
    throw err;
  }
  return data.data;
}


export async function fetchAuthenticatedUser(accessToken) {
  if (!accessToken) return null;
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  return parseJsonResponse(response);
}

export async function login({ email, password }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  return parseJsonResponse(response);
}

export async function fetchDatasets({ domain, accessToken } = {}) {
  const params = new URLSearchParams();
  if (domain) params.set('domain', domain);
  const headers = {};
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const query = params.toString();
  const response = await fetch(
    `${API_BASE_URL}/api/v1/datasets${query ? `?${query}` : ''}`,
    { headers },
  );
  return parseJsonResponse(response);
}

export async function uploadDataset({ file, domain, accessToken }) {
  const form = new FormData();
  form.append('file', file);
  form.append('domain', domain);
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${accessToken}` },
    body: form,
  });
  return parseJsonResponse(response);
}

export async function getDatasetDetail({ datasetId, accessToken, previewRows = 5 }) {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/datasets/${datasetId}?preview_rows=${previewRows}`,
    { headers: { Authorization: `Bearer ${accessToken}` } },
  );
  return parseJsonResponse(response);
}

export async function profileDataset({ datasetId, domain, target, accessToken } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/profile`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ domain, target }),
  });
  return parseJsonResponse(response);
}

export async function validateDataset({ datasetId, domain, accessToken }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/validate`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ schema_type: `${domain}_v1`, strict_mode: true }),
  });
  return parseJsonResponse(response);
}

export async function preprocessDataset({
  datasetId,
  accessToken,
  target,
  timeColumn,
  columnMapping,
}) {
  const body = {
    generate_lags: false,
    generate_forecast_features: true,
  };
  if (target) body.target = target;
  if (timeColumn) body.time_column = timeColumn;
  if (columnMapping) body.column_mapping = columnMapping;

  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/preprocess`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  return parseJsonResponse(response);
}

export async function fetchDatasetGeography({ datasetId, geoFilters, location, accessToken } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const payload = { ...(geoFilters || {}) };
  if (location) payload.location = location;
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/geography`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
  return parseJsonResponse(response);
}

export async function runForecast({
  domain,
  model,
  datasetId,
  target,
  location,
  geoFilters,
  featuresDir,
  predictionLimit = 100,
  selectionMethod,
  forceRetrain = false,
}) {
  const isDemo = !datasetId || datasetId.startsWith('demo_');
  const targetFeaturesDir = featuresDir || (isDemo ? (DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail) : undefined);
  const body = {
    domain,
    model,
    dataset_id: datasetId,
    dataset: datasetId,
    target,
    location,
    geo_filters: geoFilters,
    features_dir: targetFeaturesDir,
    prediction_limit: predictionLimit,
    force_retrain: forceRetrain,
  };
  if (selectionMethod) {
    body.selection_method = selectionMethod;
  }
  const response = await fetch(`${API_BASE_URL}/api/v1/forecasting/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJsonResponse(response);
}

export async function fetchForecastHistory({ domain, model, limit = 20 } = {}) {
  const params = new URLSearchParams();
  if (domain) params.set('domain', domain);
  if (model) params.set('model', model);
  if (limit != null) params.set('limit', String(limit));
  const query = params.toString();
  const response = await fetch(
    `${API_BASE_URL}/api/v1/forecasting/history${query ? `?${query}` : ''}`,
  );
  return parseJsonResponse(response);
}

export async function runFutureForecast({ domain, model, datasetId, target, location, geoFilters, horizon, featuresDir }) {
  const isDemo = !datasetId || datasetId.startsWith('demo_');
  const targetFeaturesDir = featuresDir || (isDemo ? (DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail) : undefined);
  const response = await fetch(`${API_BASE_URL}/api/v1/forecasting/future`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      domain,
      model,
      dataset_id: datasetId,
      dataset: datasetId,
      target,
      location,
      geo_filters: geoFilters,
      features_dir: targetFeaturesDir,
      horizon,
    }),
  });
  return parseJsonResponse(response);
}

export async function compareModels({ domain, datasetId, target, location, geoFilters, featuresDir, selectionMetric = 'rmse', models }) {
  const isDemo = !datasetId || datasetId.startsWith('demo_');
  const targetFeaturesDir = featuresDir || (isDemo ? (DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail) : undefined);
  const body = {
    domain,
    dataset_id: datasetId,
    dataset: datasetId,
    target,
    location,
    geo_filters: geoFilters,
    features_dir: targetFeaturesDir,
    selection_metric: selectionMetric,
  };
  if (models && models.length > 0) {
    body.models = models;
  }
  const response = await fetch(`${API_BASE_URL}/api/v1/forecasting/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJsonResponse(response);
}

export async function explainForecast({ domain, model, datasetId, target, location, geoFilters, featuresDir, explanationType = 'global', method = 'shap' }) {
  const isDemo = !datasetId || datasetId.startsWith('demo_');
  const targetFeaturesDir = featuresDir || (isDemo ? (DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail) : undefined);
  const response = await fetch(`${API_BASE_URL}/api/v1/explainability/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      domain,
      model,
      dataset_id: datasetId,
      dataset: datasetId,
      target,
      location,
      geo_filters: geoFilters,
      features_dir: targetFeaturesDir,
      explanation_type: explanationType,
      method,
    }),
  });
  return parseJsonResponse(response);
}

export async function fetchLlmExplanation({ domain, model, datasetId, target, location, geoFilters, featuresDir }) {
  const isDemo = !datasetId || datasetId.startsWith('demo_');
  const targetFeaturesDir = featuresDir || (isDemo ? (DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail) : undefined);
  const response = await fetch(`${API_BASE_URL}/api/v1/explainability/llm-explanation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      domain,
      model,
      dataset_id: datasetId,
      dataset: datasetId,
      target,
      location,
      geo_filters: geoFilters,
      features_dir: targetFeaturesDir,
    }),
  });
  return parseJsonResponse(response);
}



export async function fetchDashboardSummary({ recentLimit = 10 } = {}) {
  const params = new URLSearchParams();
  if (recentLimit != null) params.set('recent_limit', String(recentLimit));
  const response = await fetch(`${API_BASE_URL}/api/v1/dashboard?${params.toString()}`);
  return parseJsonResponse(response);
}

export async function fetchModelsCatalog() {
  const response = await fetch(`${API_BASE_URL}/api/v1/dashboard/models`);
  return parseJsonResponse(response);
}

export async function fetchReports({ domain, limit = 20 } = {}) {
  const params = new URLSearchParams();
  if (domain) params.set('domain', domain);
  if (limit != null) params.set('limit', String(limit));
  const query = params.toString();
  const response = await fetch(`${API_BASE_URL}/api/v1/reports${query ? `?${query}` : ''}`);
  return parseJsonResponse(response);
}

export async function fetchReport(runId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/reports/${encodeURIComponent(runId)}`);
  return parseJsonResponse(response);
}

export function getReportPdfUrl(runId) {
  return `${API_BASE_URL}/api/v1/reports/${encodeURIComponent(runId)}/pdf`;
}

export function getReportExportUrl(runId, format = 'csv') {
  return `${API_BASE_URL}/api/v1/reports/${encodeURIComponent(runId)}/export/${encodeURIComponent(format)}`;
}

export async function downloadReportPdf(runId) {
  const url = getReportPdfUrl(runId);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download PDF report (HTTP ${response.status})`);
  }
  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = `SmartXSense_Report_${runId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(downloadUrl);
}

export async function exportReportData(runId, format = 'csv') {
  const url = getReportExportUrl(runId, format);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to export report as ${format.toUpperCase()} (HTTP ${response.status})`);
  }
  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = format === 'json' ? `SmartXSense_Report_${runId}.json` : `SmartXSense_Forecast_${runId}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(downloadUrl);
}

export async function fetchSystemSettings() {
  const response = await fetch(`${API_BASE_URL}/api/v1/settings`);
  return parseJsonResponse(response);
}

export async function fetchHealthStatus() {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('Unable to reach SmartXSense health endpoint.');
  }
  if (!response.ok && !data?.data) {
    throw new Error(data?.message || `Health check failed with status ${response.status}`);
  }
  return data.data || data;
}
