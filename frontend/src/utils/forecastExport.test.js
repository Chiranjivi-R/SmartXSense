/**
 * Focused Node tests for Phase E client-side forecast export helpers.
 * Run: node --test src/utils/forecastExport.test.js
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  buildExportFilename,
  buildForecastJsonString,
  buildFutureForecastCsv,
  buildTestForecastCsv,
  sanitizeForecastJson,
  sanitizeFilenamePart,
} from './forecastExport.js';

const sampleTestForecast = {
  domain: 'retail',
  model: 'random_forest',
  target: 'sales',
  prediction_count: 2,
  test_metrics: { mae: 1.5, rmse: 2.0, mape: 0.1, r2: 0.9 },
  predictions: [
    { timestamp: '2024-01-01T00:00:00+00:00', actual: 10, predicted: 8 },
    { timestamp: '2024-01-02T00:00:00+00:00', actual: 20, predicted: 25 },
  ],
  artifact: { source: 'loaded', key: 'retail/random_forest__features_x__abc' },
  features_dir: 'C:\\Users\\secret\\features',
};

const sampleFutureForecast = {
  domain: 'retail',
  model: 'random_forest',
  target: 'sales',
  horizon: 2,
  forecast_type: 'future',
  predictions: [
    { timestamp: '2025-01-01T00:00:00+00:00', predicted: 11.5 },
    { timestamp: '2025-01-02T00:00:00+00:00', predicted: 12.25 },
  ],
};

describe('forecastExport helpers', () => {
  it('sanitizes filename parts and builds safe names', () => {
    assert.equal(sanitizeFilenamePart('Random Forest'), 'random_forest');
    assert.equal(
      buildExportFilename({ domain: 'retail', model: 'xgboost', kind: 'forecast', extension: 'csv' }),
      'smartxsense_retail_xgboost_forecast.csv',
    );
    assert.equal(
      buildExportFilename({ domain: 'retail', model: 'random_forest', kind: 'future_forecast', extension: 'json' }),
      'smartxsense_retail_random_forest_future_forecast.json',
    );
  });

  it('builds test forecast CSV with headers and absolute error', () => {
    const csv = buildTestForecastCsv(sampleTestForecast);
    const lines = csv.trim().split('\n');
    assert.equal(lines[0], 'timestamp,actual,predicted,absolute_error');
    assert.ok(lines[1].includes('2024-01-01'));
    assert.ok(lines[1].includes(',10,8,2'));
    assert.ok(lines[2].includes(',20,25,5'));
  });

  it('rejects empty test forecast data', () => {
    assert.throws(() => buildTestForecastCsv(null), /No test forecast/);
    assert.throws(() => buildTestForecastCsv({ predictions: null }), /No test forecast/);
  });

  it('builds future CSV without actual column', () => {
    const csv = buildFutureForecastCsv(sampleFutureForecast);
    const lines = csv.trim().split('\n');
    assert.equal(lines[0], 'timestamp,predicted,horizon_index');
    assert.equal(lines[0].includes('actual'), false);
    assert.ok(lines[1].includes('11.5'));
    assert.ok(lines[1].endsWith(',1'));
    assert.ok(!lines.some((line) => /^timestamp,actual/.test(line)));
  });

  it('omits actual from future CSV even if a row incorrectly includes it', () => {
    const dirty = {
      ...sampleFutureForecast,
      predictions: [{ timestamp: '2025-01-01', predicted: 9, actual: 999 }],
    };
    const csv = buildFutureForecastCsv(dirty);
    assert.ok(!csv.split('\n')[0].includes('actual'));
    assert.ok(!csv.includes(',999'));
  });

  it('JSON export preserves forecast fields and strips path-like keys', () => {
    const parsed = JSON.parse(buildForecastJsonString(sampleTestForecast));
    assert.equal(parsed.domain, 'retail');
    assert.equal(parsed.model, 'random_forest');
    assert.equal(parsed.predictions.length, 2);
    assert.equal(parsed.test_metrics.rmse, 2.0);
    assert.equal(Object.prototype.hasOwnProperty.call(parsed, 'features_dir'), false);

    const scrubbed = sanitizeForecastJson({
      domain: 'retail',
      path: '/tmp/secret',
      nested: { absolute_path: 'C:\\x\\y' },
    });
    assert.equal(scrubbed.domain, 'retail');
    assert.equal(Object.prototype.hasOwnProperty.call(scrubbed, 'path'), false);
    assert.equal(Object.prototype.hasOwnProperty.call(scrubbed.nested, 'absolute_path'), false);
  });

  it('rejects empty JSON export input', () => {
    assert.throws(() => buildForecastJsonString(null), /No forecast data/);
  });
});
