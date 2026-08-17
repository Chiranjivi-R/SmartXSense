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
  retail: ['random_forest', 'xgboost'],
  electricity: ['random_forest', 'xgboost', 'arima'],
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
    throw new Error((data.message || `Request failed with status ${response.status}`) + details);
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

export async function preprocessDataset({ datasetId, accessToken }) {
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/preprocess`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      generate_lags: false,
      generate_forecast_features: true,
    }),
  });
  return parseJsonResponse(response);
}

export async function runForecast({
  domain,
  model,
  featuresDir,
  predictionLimit = 100,
  selectionMethod,
  forceRetrain = false,
}) {
  const targetFeaturesDir = featuresDir || DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail;
  const body = {
    domain,
    model,
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

export async function runFutureForecast({ domain, model, horizon, featuresDir }) {
  const targetFeaturesDir = featuresDir || DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail;
  const response = await fetch(`${API_BASE_URL}/api/v1/forecasting/future`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      domain,
      model,
      features_dir: targetFeaturesDir,
      horizon,
    }),
  });
  return parseJsonResponse(response);
}

export async function compareModels({ domain, featuresDir, selectionMetric = 'rmse', models }) {
  const targetFeaturesDir = featuresDir || DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail;
  const body = {
    domain,
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

export async function explainForecast({ domain, model, featuresDir, explanationType = 'global', method = 'shap' }) {
  const targetFeaturesDir = featuresDir || DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail;
  const response = await fetch(`${API_BASE_URL}/api/v1/explainability/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      domain,
      model,
      features_dir: targetFeaturesDir,
      explanation_type: explanationType,
      method,
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
