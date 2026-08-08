/**
 * Client-side forecast export helpers (Phase E).
 * Pure serialization — no API calls, no filesystem path exposure.
 */

const PATH_LIKE_KEYS = new Set([
  'features_dir',
  'path',
  'filepath',
  'file_path',
  'absolute_path',
  'artifacts_root',
  'history_root',
]);

/**
 * Sanitize a token for use in a download filename.
 */
export function sanitizeFilenamePart(value, fallback = 'export') {
  const cleaned = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64);
  return cleaned || fallback;
}

/**
 * Build a safe download filename.
 * @param {{ domain?: string, model?: string, kind?: 'forecast' | 'future_forecast', extension: 'csv' | 'json' }} opts
 */
export function buildExportFilename({ domain, model, kind = 'forecast', extension }) {
  const d = sanitizeFilenamePart(domain, 'domain');
  const m = sanitizeFilenamePart(model, 'model');
  const k = kind === 'future_forecast' ? 'future_forecast' : 'forecast';
  const ext = extension === 'json' ? 'json' : 'csv';
  return `smartxsense_${d}_${m}_${k}.${ext}`;
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

/**
 * Build CSV for a test-evaluation forecast result.
 * Columns: timestamp,actual,predicted,absolute_error
 */
export function buildTestForecastCsv(forecastData) {
  if (!forecastData || !Array.isArray(forecastData.predictions)) {
    throw new Error('No test forecast predictions available to export.');
  }

  const lines = ['timestamp,actual,predicted,absolute_error'];
  for (const row of forecastData.predictions) {
    const actual = Number(row.actual);
    const predicted = Number(row.predicted);
    const absErr =
      Number.isFinite(actual) && Number.isFinite(predicted)
        ? Math.abs(actual - predicted)
        : '';
    lines.push(
      [
        csvEscape(row.timestamp ?? ''),
        csvEscape(Number.isFinite(actual) ? actual : row.actual ?? ''),
        csvEscape(Number.isFinite(predicted) ? predicted : row.predicted ?? ''),
        csvEscape(absErr),
      ].join(','),
    );
  }
  return `${lines.join('\n')}\n`;
}

/**
 * Build CSV for a future forecast result.
 * Columns: timestamp,predicted,horizon_index
 * Never invents an actual column.
 */
export function buildFutureForecastCsv(futureData) {
  if (!futureData || !Array.isArray(futureData.predictions)) {
    throw new Error('No future forecast predictions available to export.');
  }

  const lines = ['timestamp,predicted,horizon_index'];
  futureData.predictions.forEach((row, index) => {
    if (Object.prototype.hasOwnProperty.call(row, 'actual')) {
      // Defensive: never export fabricated actuals even if present.
      // Intentionally omit actual from output columns.
    }
    const predicted = Number(row.predicted);
    lines.push(
      [
        csvEscape(row.timestamp ?? ''),
        csvEscape(Number.isFinite(predicted) ? predicted : row.predicted ?? ''),
        csvEscape(index + 1),
      ].join(','),
    );
  });
  return `${lines.join('\n')}\n`;
}

/**
 * Remove absolute/path-like keys from a deep clone of forecast JSON.
 */
export function sanitizeForecastJson(data) {
  if (data === null || data === undefined) {
    throw new Error('No forecast data available to export.');
  }

  const clone = structuredClone
    ? structuredClone(data)
    : JSON.parse(JSON.stringify(data));

  const scrub = (node) => {
    if (Array.isArray(node)) {
      node.forEach(scrub);
      return;
    }
    if (!node || typeof node !== 'object') return;
    for (const key of Object.keys(node)) {
      if (PATH_LIKE_KEYS.has(key)) {
        delete node[key];
        continue;
      }
      const value = node[key];
      if (typeof value === 'string') {
        // Drop Windows/Unix absolute paths if somehow present as free-form strings.
        if (/^[A-Za-z]:[\\/]/.test(value) || value.startsWith('/Users/') || value.startsWith('/home/')) {
          delete node[key];
          continue;
        }
      }
      scrub(value);
    }
  };

  scrub(clone);
  return clone;
}

export function buildForecastJsonString(data) {
  return `${JSON.stringify(sanitizeForecastJson(data), null, 2)}\n`;
}

/**
 * Trigger a browser download from a Blob.
 */
export function downloadTextFile(filename, content, mimeType) {
  if (typeof document === 'undefined') {
    throw new Error('Browser download is only available in the UI.');
  }
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function exportTestForecastCsv(forecastData) {
  const csv = buildTestForecastCsv(forecastData);
  const filename = buildExportFilename({
    domain: forecastData.domain,
    model: forecastData.model,
    kind: 'forecast',
    extension: 'csv',
  });
  downloadTextFile(filename, csv, 'text/csv;charset=utf-8');
  return filename;
}

export function exportTestForecastJson(forecastData) {
  const json = buildForecastJsonString(forecastData);
  const filename = buildExportFilename({
    domain: forecastData.domain,
    model: forecastData.model,
    kind: 'forecast',
    extension: 'json',
  });
  downloadTextFile(filename, json, 'application/json;charset=utf-8');
  return filename;
}

export function exportFutureForecastCsv(futureData) {
  const csv = buildFutureForecastCsv(futureData);
  const filename = buildExportFilename({
    domain: futureData.domain,
    model: futureData.model,
    kind: 'future_forecast',
    extension: 'csv',
  });
  downloadTextFile(filename, csv, 'text/csv;charset=utf-8');
  return filename;
}

export function exportFutureForecastJson(futureData) {
  const json = buildForecastJsonString(futureData);
  const filename = buildExportFilename({
    domain: futureData.domain,
    model: futureData.model,
    kind: 'future_forecast',
    extension: 'json',
  });
  downloadTextFile(filename, json, 'application/json;charset=utf-8');
  return filename;
}
