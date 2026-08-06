/**
 * API client service for SmartXSense ML Forecasting and Explainability endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export const DOMAIN_FEATURE_ARTIFACTS = {
  retail: 'ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4',
  electricity: 'ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25',
};

/**
 * Call forecast endpoint
 * @param {Object} params
 * @param {string} params.domain - 'retail' or 'electricity'
 * @param {string} params.model - 'random_forest', 'xgboost', 'hybrid', 'arima', 'naive', 'seasonal_naive'
 * @param {string} [params.featuresDir] - Path to features directory (defaults to domain artifact)
 * @param {number} [params.predictionLimit=100] - Prediction limit for response
 */
export async function runForecast({ domain, model, featuresDir, predictionLimit = 100 }) {
  const targetFeaturesDir = featuresDir || DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail;

  const response = await fetch(`${API_BASE_URL}/api/v1/forecasting/forecast`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      domain,
      model,
      features_dir: targetFeaturesDir,
      prediction_limit: predictionLimit,
    }),
  });

  const data = await response.json();
  if (!response.ok || data.status === 'error') {
    throw new Error(data.message || `Forecast request failed with status ${response.status}`);
  }
  return data.data;
}

/**
 * Call model comparison endpoint
 * @param {Object} params
 * @param {string} params.domain - 'retail' or 'electricity'
 * @param {string} [params.featuresDir]
 * @param {string} [params.selectionMetric='rmse']
 * @param {Array<string>} [params.models]
 */
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
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok || data.status === 'error') {
    throw new Error(data.message || `Model comparison request failed with status ${response.status}`);
  }
  return data.data;
}

/**
 * Call explainability endpoint
 * @param {Object} params
 * @param {string} params.domain - 'retail' or 'electricity'
 * @param {string} params.model - 'random_forest' or 'xgboost'
 * @param {string} [params.featuresDir]
 * @param {string} [params.explanationType='global']
 * @param {string} [params.method='shap']
 */
export async function explainForecast({ domain, model, featuresDir, explanationType = 'global', method = 'shap' }) {
  const targetFeaturesDir = featuresDir || DOMAIN_FEATURE_ARTIFACTS[domain] || DOMAIN_FEATURE_ARTIFACTS.retail;

  const response = await fetch(`${API_BASE_URL}/api/v1/explainability/explain`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      domain,
      model,
      features_dir: targetFeaturesDir,
      explanation_type: explanationType,
      method,
    }),
  });

  const data = await response.json();
  if (!response.ok || data.status === 'error') {
    throw new Error(data.message || `Explainability request failed with status ${response.status}`);
  }
  return data.data;
}
