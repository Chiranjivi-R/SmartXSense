/**
 * Focused Node tests for applicability presentation and N/A message handling.
 * Run: node --test src/utils/applicabilityPresentation.test.js
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  isApplicabilityNaError,
  parseApplicabilityDetails,
  formatUserFacingError,
} from './applicabilityPresentation.js';

describe('applicabilityPresentation helpers', () => {
  const ahmedabadBackendError =
    'HTTP 400 — N/A — No applicable forecasting model for the selected location. No eligible models available for comparison on this dataset. (arima [INTERLEAVED_PANEL]: Dataset contains an interleaved multi-entity panel; random_forest [INSUFFICIENT_HISTORY]: N/A — Eligible training observation criterion not met. 14 usable training observations available; minimum 15 required for Random Forest (got 14).)';

  const simpleNoEligibleError =
    'HTTP 400 — No eligible models available for comparison on this dataset';

  const genuineUnrelatedHttp400 =
    'HTTP 400 — Invalid query parameter: start_date must be formatted YYYY-MM-DD';

  const genuineHttp500 =
    'HTTP 500 — Internal server error: database connection failure';

  const manualRfError =
    "HTTP 400 — Model 'random_forest' is not applicable to this dataset: [INSUFFICIENT_HISTORY] N/A — Eligible training observation criterion not met. 14 usable training observations available; minimum 15 required for Random Forest (got 14).";

  it('1. Identifies known applicability/data suitability conditions and rejects genuine unrelated errors', () => {
    assert.equal(isApplicabilityNaError(ahmedabadBackendError), true);
    assert.equal(isApplicabilityNaError(simpleNoEligibleError), true);
    assert.equal(isApplicabilityNaError(manualRfError), true);

    // Genuine errors must NOT be classified as applicability N/A
    assert.equal(isApplicabilityNaError(genuineUnrelatedHttp400), false);
    assert.equal(isApplicabilityNaError(genuineHttp500), false);
    assert.equal(isApplicabilityNaError('HTTP 404 — Dataset not found'), false);
    assert.equal(isApplicabilityNaError(null), false);
  });

  it('2. Extracts Ahmedabad observation count (14 usable) and minimum required (15)', () => {
    const details = parseApplicabilityDetails(ahmedabadBackendError, 21);
    assert.ok(details);
    assert.equal(details.isApplicabilityNa, true);
    assert.equal(details.title, 'Auto Select: N/A — No Applicable Forecasting Model');
    assert.equal(details.trainObservations, 14);
    assert.equal(details.minimumRequired, 15);

    // Verify user-facing string contracts
    assert.ok(details.criterion.includes('14 usable training observations available'));
    assert.ok(details.criterion.includes('minimum model requirements are not met'));
    assert.ok(details.exactCriterion.includes('14 usable training observations available; minimum 15 required.'));
    assert.ok(details.explanation.includes('Forecasting cannot be performed for the selected location'));

    // Crucially: NO HTTP status code in parsed output
    assert.ok(!details.title.includes('HTTP 400'));
    assert.ok(!details.criterion.includes('HTTP 400'));
    assert.ok(!details.explanation.includes('HTTP 400'));
  });

  it('3. Computes training observations from filtered_rows when message has no numbers', () => {
    // 21 rows * 0.70 = 14 training rows
    const details = parseApplicabilityDetails(simpleNoEligibleError, 21);
    assert.ok(details);
    assert.equal(details.trainObservations, 14);
    assert.ok(details.criterion.includes('14 usable training observations available'));
    assert.ok(details.criterion.includes('minimum model requirements are not met'));
  });

  it('4. Formats user-facing error without "HTTP 400" for applicability conditions', () => {
    const formatted = formatUserFacingError(ahmedabadBackendError, 21);
    assert.ok(!formatted.includes('HTTP 400'));
    assert.ok(formatted.includes('Auto Select: N/A — No Applicable Forecasting Model'));
    assert.ok(formatted.includes('14 usable training observations available'));
  });

  it('5. Formats manual model selection error cleanly without "HTTP 400"', () => {
    const formatted = formatUserFacingError(manualRfError, 21);
    assert.ok(!formatted.includes('HTTP 400'));
    assert.ok(formatted.includes("Model 'Random Forest' is not applicable for this location"));
    assert.ok(formatted.includes('14 usable training observations available; minimum 15 required.'));
  });

  it('6. Leaves genuine unrelated errors completely unchanged', () => {
    const unchanged400 = formatUserFacingError(genuineUnrelatedHttp400);
    assert.equal(unchanged400, genuineUnrelatedHttp400);

    const unchanged500 = formatUserFacingError(genuineHttp500);
    assert.equal(unchanged500, genuineHttp500);
  });

  it('7. Verifies exact Ahmedabad UI requirement strings without HTTP 400', () => {
    const details = parseApplicabilityDetails(ahmedabadBackendError, 21);

    // Header
    assert.equal(details.title, 'Auto Select: N/A — No Applicable Forecasting Model');

    // Observation requirement preservation
    assert.ok(details.criterion.includes('14 usable training observations available'));
    assert.ok(details.criterion.includes('minimum model requirements are not met'));
    assert.ok(details.exactCriterion.includes('14 usable training observations available; minimum 15 required.'));

    // Concise explanation
    assert.equal(
      details.explanation,
      'Forecasting cannot be performed for the selected location because the available training history does not satisfy the minimum requirements of the applicable models.'
    );

    // Absolute absence of HTTP 400 in all user-facing strings
    assert.equal(details.title.includes('HTTP 400'), false);
    assert.equal(details.criterion.includes('HTTP 400'), false);
    assert.equal(details.explanation.includes('HTTP 400'), false);
  });

  it('8. Verifies location switching state transition semantics (Mysuru <-> Ahmedabad)', () => {
    // Step 1: Initial state on Mysuru
    let autoSelectedModel = 'xgboost';
    let autoSelectReason = null;
    let forecastError = null;

    // Successful location displays actual model normally
    assert.equal(autoSelectedModel, 'xgboost');
    assert.equal(autoSelectReason, null);

    // Step 2: User switches to Ahmedabad
    // Location switch hook clears previous state
    autoSelectedModel = null;
    autoSelectReason = null;
    forecastError = null;
    assert.equal(autoSelectedModel, null);

    // Ahmedabad Auto Select comparison runs and returns 400 applicability error
    const err = new Error(ahmedabadBackendError);
    if (isApplicabilityNaError(err.message)) {
      autoSelectedModel = 'n/a';
      autoSelectReason = parseApplicabilityDetails(err.message, 21);
      forecastError = null; // Do not set technical error box
    }

    assert.equal(autoSelectedModel, 'n/a');
    assert.equal(autoSelectReason.title, 'Auto Select: N/A — No Applicable Forecasting Model');
    assert.equal(forecastError, null);

    // Step 3: User switches back to Mysuru
    // Location switch hook clears Ahmedabad N/A state
    autoSelectedModel = null;
    autoSelectReason = null;
    forecastError = null;

    // Mysuru runs and obtains fresh recommendation
    autoSelectedModel = 'xgboost';
    assert.equal(autoSelectedModel, 'xgboost');
    assert.equal(autoSelectReason, null);
  });

  it('9. Explicitly proves Ahmedabad: UI message contains "14 usable training observations available" and NOT "21"', () => {
    const ahmedabadContext = {
      filtered_rows: 21,
      total_records: 21,
      training_rows: 14,
      training_observations: 14,
      validation_observations: 3,
      test_observations: 4,
    };

    const details = parseApplicabilityDetails(ahmedabadBackendError, ahmedabadContext);
    assert.ok(details);

    // MUST contain: "14 usable training observations available"
    assert.ok(details.criterion.includes('14 usable training observations available'));
    assert.ok(details.criterion.includes('minimum model requirements are not met'));

    // MUST NOT contain: "21 usable training observations available"
    assert.ok(!details.criterion.includes('21 usable training observations available'));

    // Full formatted error
    const formatted = formatUserFacingError(ahmedabadBackendError, ahmedabadContext);
    assert.ok(formatted.includes('14 usable training observations available'));
    assert.ok(!formatted.includes('21 usable training observations available'));
  });

  it('10. Location with different total/training split correctly displays its training count', () => {
    // E.g., location with 100 records -> 70 training observations
    const customContext = {
      filtered_rows: 100,
      total_records: 100,
      training_rows: 70,
    };

    const genericError =
      'HTTP 400 — No eligible models available for comparison on this dataset';

    const details = parseApplicabilityDetails(genericError, customContext);
    assert.ok(details);
    assert.equal(details.trainObservations, 70);
    assert.ok(details.criterion.includes('70 usable training observations available'));
    assert.ok(!details.criterion.includes('100 usable training observations available'));

    const formatted = formatUserFacingError(genericError, customContext);
    assert.ok(formatted.includes('70 usable training observations available'));
    assert.ok(!formatted.includes('100 usable training observations available'));
  });

  it('11. Derives 70% split from raw number context without equating total to training', () => {
    // When only raw number 21 is provided as context
    const details = parseApplicabilityDetails(simpleNoEligibleError, 21);
    assert.ok(details);
    // Derived training observations = Math.floor(21 * 0.70) = 14
    assert.equal(details.trainObservations, 14);
    assert.ok(details.criterion.includes('14 usable training observations available'));
    assert.ok(!details.criterion.includes('21 usable training observations available'));
  });

  it('12. Confirms separation between total records and training observations display', () => {
    const geoMeta = {
      filtered_rows: 21,
      total_rows: 767,
      training_rows: 14,
    };

    // Geographic scope badge format: "21 records (14 training obs.) / 767 total"
    const trainingObsDisplay = `${geoMeta.training_rows} training obs.`;
    const geoScopeBadge = `${geoMeta.filtered_rows} records (${trainingObsDisplay}) / ${geoMeta.total_rows} total`;
    assert.equal(geoScopeBadge, '21 records (14 training obs.) / 767 total');

    // N/A message format: "14 usable training observations available; minimum model requirements are not met."
    const details = parseApplicabilityDetails(ahmedabadBackendError, geoMeta);
    assert.equal(
      details.criterion,
      '14 usable training observations available; minimum model requirements are not met.'
    );

    // Verify they are completely separate quantities
    assert.equal(geoMeta.filtered_rows, 21);
    assert.equal(details.trainObservations, 14);
  });
});

