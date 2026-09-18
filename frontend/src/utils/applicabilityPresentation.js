/**
 * Applicability and N/A presentation helpers for SmartXSense frontend.
 * 
 * Transforms technical backend HTTP 400 applicability errors into clean,
 * user-friendly N/A informational states without raw HTTP status codes or internal API wording.
 */

const APPLICABILITY_KEYWORDS = [
  'no eligible models',
  'no applicable forecasting model',
  'insufficient forecasting history',
  'eligible training observation criterion not met',
  'insufficient_history',
  'insufficient_sequence_history',
  'fewer_than_two_components',
  'interleaved_panel',
  'unsuitable time-series structure',
  'minimum 10 required',
  'insufficient observations',
  'dataset has only',
  'too small for time-series split',
  'unique timestamps',
  'at least 3 are required',
  'no records found for the selected location',
];

/**
 * Determine if an error message represents a known forecasting applicability / data suitability limitation.
 * Other genuine errors (HTTP 500, 404, 401, invalid input, etc.) return false.
 *
 * @param {string} errMessage
 * @returns {boolean}
 */
export function isApplicabilityNaError(errMessage) {
  if (!errMessage || typeof errMessage !== 'string') return false;
  const lower = errMessage.toLowerCase();
  return APPLICABILITY_KEYWORDS.some((kw) => lower.includes(kw));
}

/**
 * Parse structured details from an applicability error message.
 *
 * @param {string} errMessage - Raw error message (may contain "HTTP 400 — ...")
 * @param {number|null} filteredRows - Current location's filtered row count if known
 * @returns {object|null}
 */
/**
 * Parse structured details from an applicability error message.
 *
 * @param {string} errMessage - Raw error message (may contain "HTTP 400 — ...")
 * @param {object|number|null} context - Context object (e.g. geoMeta) or total row count
 * @returns {object|null}
 */
export function parseApplicabilityDetails(errMessage, context = null) {
  if (!errMessage || !isApplicabilityNaError(errMessage)) {
    return null;
  }

  let trainObs = null;
  let minRequired = null;
  let specificModel = null;

  const isAutoSelect =
    errMessage.toLowerCase().includes('no eligible models') ||
    errMessage.toLowerCase().includes('for comparison') ||
    errMessage.toLowerCase().includes('no applicable forecasting model for the selected location');

  if (!isAutoSelect) {
    const modelMatch = errMessage.match(/Model\s+'([^']+)'\s+is not applicable/i);
    if (modelMatch) {
      specificModel = modelMatch[1].trim();
    }
  }

  // 1. Usable training observations extraction strictly adhering to Priority 1-5:

  // Priority 1: explicit API training_observations
  if (context && typeof context === 'object') {
    if (typeof context.training_observations === 'number') {
      trainObs = context.training_observations;
    } else if (typeof context.trainingObservations === 'number') {
      trainObs = context.trainingObservations;
    }
  }

  // Priority 2: explicit API training_rows
  if (trainObs == null && context && typeof context === 'object') {
    if (typeof context.training_rows === 'number') {
      trainObs = context.training_rows;
    } else if (typeof context.trainingRows === 'number') {
      trainObs = context.trainingRows;
    }
  }

  // Priority 3: canonical split metadata
  if (trainObs == null && context && typeof context === 'object') {
    const splitMeta = context.split_metadata || context.canonical_split || context.splitMetadata || context.canonicalSplit;
    if (splitMeta && typeof splitMeta === 'object') {
      const splitVal = splitMeta.training ?? splitMeta.train ?? splitMeta.training_observations ?? splitMeta.training_rows;
      if (typeof splitVal === 'number') {
        trainObs = splitVal;
      }
    }
  }

  // Priority 4: explicit count in backend message
  if (trainObs == null && typeof errMessage === 'string') {
    const trainMatch =
      errMessage.match(/(\d+)\s+usable training observations? available/i) ||
      errMessage.match(/(\d+)\s+usable training observations?/i);
    if (trainMatch) {
      trainObs = parseInt(trainMatch[1], 10);
    }
  }

  // Priority 5: canonical split calculation only as a fallback (70% chronological split)
  // NEVER use filtered_rows or total_records directly as training observations!
  if (trainObs == null) {
    let totalRecords = null;
    if (context && typeof context === 'object') {
      if (typeof context.filtered_rows === 'number') {
        totalRecords = context.filtered_rows;
      } else if (typeof context.total_records === 'number') {
        totalRecords = context.total_records;
      } else if (typeof context.totalRows === 'number') {
        totalRecords = context.totalRows;
      }
    } else if (typeof context === 'number' && context > 0) {
      totalRecords = context;
    }

    if (totalRecords != null && totalRecords > 0) {
      trainObs = Math.floor(totalRecords * 0.70);
    }
  }

  // Fallback pattern matching on sequence lookback history if still null
  if (trainObs == null && typeof errMessage === 'string') {
    const gotMatch =
      errMessage.match(/got\s+(\d+)\s*(?:rows|training observations|\))/i) ||
      errMessage.match(/\(got\s+(\d+)\)/i) ||
      errMessage.match(/got\s+(\d+)/i);
    if (gotMatch) {
      trainObs = parseInt(gotMatch[1], 10);
    }
  }

  // 2. Minimum required observations extraction
  const minMatch = errMessage.match(/minimum\s+(\d+)\s+required/i) ||
                   errMessage.match(/at least\s+(\d+)\s+training observations/i) ||
                   errMessage.match(/requires >\s*(\d+)/i);
  if (minMatch) {
    minRequired = parseInt(minMatch[1], 10);
  }

  const splitMatch = errMessage.match(/(?:has only|has)\s+(\d+)\s+unique timestamps?;\s*at least\s+(\d+)\s+are required/i);
  if (splitMatch) {
    if (trainObs === null) trainObs = parseInt(splitMatch[1], 10);
    if (minRequired === null) minRequired = parseInt(splitMatch[2], 10);
  }

  // Build user-friendly criterion string
  let criterion = '';
  let exactCriterion = '';
  if (splitMatch) {
    criterion = `Available data has only ${splitMatch[1]} unique timestamp(s); minimum ${splitMatch[2]} required for chronological splitting.`;
    exactCriterion = `Selected scope has only ${splitMatch[1]} unique timestamp(s); minimum ${splitMatch[2]} required for chronological splitting.`;
  } else if (isAutoSelect) {
    if (trainObs !== null) {
      criterion = `${trainObs} usable training observations available; minimum model requirements are not met.`;
      exactCriterion = minRequired !== null
        ? `${trainObs} usable training observations available; minimum ${minRequired} required.`
        : criterion;
    } else {
      criterion = 'Available training history does not satisfy the minimum requirements of the applicable models.';
      exactCriterion = criterion;
    }
  } else {
    // Specific model selection
    if (trainObs !== null && minRequired !== null) {
      exactCriterion = `${trainObs} usable training observations available; minimum ${minRequired} required.`;
      criterion = exactCriterion;
    } else if (trainObs !== null) {
      exactCriterion = `${trainObs} usable training observations available; minimum model requirements are not met.`;
      criterion = exactCriterion;
    } else {
      criterion = 'Available training history does not satisfy the minimum requirements of the applicable models.';
      exactCriterion = criterion;
    }
  }

  const explanation =
    'Forecasting cannot be performed for the selected location because the available training history does not satisfy the minimum requirements of the applicable models.';

  return {
    isApplicabilityNa: true,
    title: 'Auto Select: N/A — No Applicable Forecasting Model',
    header: 'Auto Select: N/A — No Applicable Forecasting Model',
    criterion,
    exactCriterion,
    explanation,
    trainObservations: trainObs,
    training_observations: trainObs,
    minimumRequired: minRequired,
    specificModel,
  };
}

/**
 * Format any general error message for user-facing display.
 * If the message represents a known applicability limitation, strips technical HTTP status
 * codes and returns a clean N/A description. Unrelated errors remain untouched.
 *
 * @param {string} errMessage
 * @param {object|number|null} context
 * @returns {string}
 */
export function formatUserFacingError(errMessage, context = null) {
  if (!errMessage || typeof errMessage !== 'string') return '';
  if (!isApplicabilityNaError(errMessage)) {
    return errMessage;
  }

  const details = parseApplicabilityDetails(errMessage, context);
  if (!details) return errMessage;

  if (details.specificModel) {
    const formattedModel = details.specificModel
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
    const crit = details.exactCriterion || details.criterion;
    return `N/A — Model '${formattedModel}' is not applicable for this location: ${crit}`;
  }

  return `Auto Select: N/A — No Applicable Forecasting Model. ${details.criterion}`;
}
