import React, { useEffect, useState } from 'react';
import {
  downloadReportPdf,
  exportReportData,
  fetchReport,
  fetchReports,
} from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  primaryGlow: 'rgba(0, 212, 255, 0.15)',
  accent: '#FF6B6B',
  success: '#00E5A0',
  warning: '#FFB800',
  bg: {
    dark: '#060B1A',
    card: '#0D1530',
    cardHeader: '#111B3D',
    border: '#1E2D5A',
    innerBox: '#080E24',
  },
  text: {
    primary: '#E8F0FF',
    secondary: '#8899CC',
    muted: '#4A5B8A',
  },
};

const formatMape = (val) => {
  if (val == null || Number.isNaN(Number(val))) return '—';
  const num = Number(val);
  const pct = num < 50 ? num * 100 : num;
  return `${pct.toFixed(2)}%`;
};

const formatNumber = (val, decimals = 2) => {
  if (val == null || Number.isNaN(Number(val))) return '—';
  return Number(val).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

export default function ReportsPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reports, setReports] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Filters
  const [domainFilter, setDomainFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');

  // Export action state
  const [actionLoading, setActionLoading] = useState(null);
  const [actionMessage, setActionMessage] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchReports({ limit: 50 });
      setReports(res.reports || []);
      if (!selectedId && res.reports?.length > 0) {
        openReport(res.reports[0].run_id);
      }
    } catch (err) {
      setError(err.message || 'Unable to load reports.');
      setReports([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openReport = async (runId) => {
    setSelectedId(runId);
    setDetailLoading(true);
    setDetailError(null);
    setActionMessage(null);
    try {
      const data = await fetchReport(runId);
      setDetail(data);
    } catch (err) {
      setDetail(null);
      setDetailError(err.message || 'Unable to load report detail.');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDownloadPdf = async (runId) => {
    setActionLoading('pdf');
    setActionMessage(null);
    try {
      await downloadReportPdf(runId);
      setActionMessage({ type: 'success', text: 'PDF audit report downloaded successfully.' });
    } catch (err) {
      setActionMessage({ type: 'error', text: err.message || 'Failed to download PDF.' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleExportData = async (runId, format) => {
    setActionLoading(format);
    setActionMessage(null);
    try {
      await exportReportData(runId, format);
      setActionMessage({ type: 'success', text: `${format.toUpperCase()} export downloaded successfully.` });
    } catch (err) {
      setActionMessage({ type: 'error', text: err.message || `Failed to export ${format.toUpperCase()}.` });
    } finally {
      setActionLoading(null);
    }
  };

  const filteredReports = reports.filter((r) => {
    if (domainFilter !== 'all' && r.domain?.toLowerCase() !== domainFilter.toLowerCase()) return false;
    if (typeFilter !== 'all' && r.forecast_type !== typeFilter) return false;
    return true;
  });

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', color: COLORS.text.primary }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, letterSpacing: '-0.02em', color: COLORS.text.primary }}>
            Forecast Reports &amp; Audit Log
          </h2>
          <p style={{ margin: '6px 0 0', fontSize: 13, color: COLORS.text.secondary }}>
            Authoritative, reproducible records of all forecast runs — with held-out test evaluation, validation benchmarks, and PDF audit generation.
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          style={{
            background: COLORS.bg.card,
            border: `1px solid ${COLORS.bg.border}`,
            color: COLORS.primary,
            borderRadius: 10,
            padding: '8px 16px',
            fontSize: 13,
            fontWeight: 700,
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {loading ? 'Refreshing…' : '↻ Refresh Reports'}
        </button>
      </div>

      {error && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: 14, color: COLORS.accent, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {/* Main Grid: Left Runs Sidebar / Right Detail Panel */}
      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 20, alignItems: 'start' }}>
        {/* Left Column: Filter & Runs List */}
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 16 }}>
          {/* Domain Filters */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em', marginBottom: 6 }}>
              DOMAIN FILTER
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {['all', 'retail', 'electricity'].map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => setDomainFilter(d)}
                  style={{
                    flex: 1,
                    background: domainFilter === d ? COLORS.primary : 'transparent',
                    color: domainFilter === d ? COLORS.bg.dark : COLORS.text.secondary,
                    border: `1px solid ${domainFilter === d ? COLORS.primary : COLORS.bg.border}`,
                    borderRadius: 8,
                    padding: '5px 8px',
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: 'pointer',
                    textTransform: 'capitalize',
                  }}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Type Filters */}
          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em', marginBottom: 6 }}>
              FORECAST TYPE
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {[
                { key: 'all', label: 'All' },
                { key: 'test_evaluation', label: 'Test Eval' },
                { key: 'future', label: 'Future' },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setTypeFilter(key)}
                  style={{
                    flex: 1,
                    background: typeFilter === key ? `${COLORS.primary}22` : 'transparent',
                    color: typeFilter === key ? COLORS.primary : COLORS.text.secondary,
                    border: `1px solid ${typeFilter === key ? COLORS.primary : COLORS.bg.border}`,
                    borderRadius: 8,
                    padding: '5px 8px',
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: 'pointer',
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.text.muted, marginBottom: 8 }}>
            Showing {filteredReports.length} of {reports.length} runs
          </div>

          {/* Runs Scroll List */}
          <div style={{ maxHeight: 'calc(100vh - 280px)', overflowY: 'auto', paddingRight: 4 }}>
            {loading && reports.length === 0 && (
              <div style={{ color: COLORS.text.secondary, padding: 20, textAlign: 'center' }}>Loading reports…</div>
            )}
            {!loading && filteredReports.length === 0 && (
              <div style={{ color: COLORS.text.muted, padding: 20, textAlign: 'center', fontSize: 13 }}>
                No forecast runs match the active filters.
              </div>
            )}
            {filteredReports.map((report) => {
              const isSelected = selectedId === report.run_id;
              const isFuture = report.forecast_type === 'future';
              const isElec = report.domain === 'electricity';

              return (
                <button
                  key={report.run_id}
                  type="button"
                  onClick={() => openReport(report.run_id)}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    background: isSelected ? `${COLORS.primary}18` : COLORS.bg.innerBox,
                    border: `1px solid ${isSelected ? COLORS.primary : COLORS.bg.border}`,
                    borderRadius: 10,
                    padding: 12,
                    marginBottom: 8,
                    cursor: 'pointer',
                    color: COLORS.text.primary,
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 800,
                        padding: '2px 6px',
                        borderRadius: 4,
                        background: isElec ? '#00D4FF1F' : '#00E5A01F',
                        color: isElec ? COLORS.primary : COLORS.success,
                        textTransform: 'uppercase',
                        letterSpacing: '0.04em',
                      }}
                    >
                      {report.domain}
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 700,
                        padding: '2px 6px',
                        borderRadius: 4,
                        background: isFuture ? '#FFB8001F' : '#8899CC1F',
                        color: isFuture ? COLORS.warning : COLORS.text.secondary,
                      }}
                    >
                      {isFuture ? 'Future' : 'Test Eval'}
                    </span>
                  </div>

                  <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'capitalize', color: COLORS.text.primary }}>
                    {String(report.model || '').replace(/_/g, ' ')}
                  </div>

                  <div style={{ fontSize: 11, color: COLORS.text.secondary, marginTop: 4 }}>
                    {report.timestamp ? new Date(report.timestamp).toLocaleString() : '—'}
                  </div>

                  <div style={{ fontSize: 11, color: COLORS.text.muted, marginTop: 4, display: 'flex', gap: 10 }}>
                    {isFuture ? (
                      <span>Horizon: {report.horizon ?? '—'} steps</span>
                    ) : (
                      <>
                        <span>RMSE: {formatNumber(report.metrics?.rmse)}</span>
                        <span>MAPE: {formatMape(report.metrics?.mape)}</span>
                      </>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Comprehensive Report Detail View */}
        <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 20 }}>
          {!selectedId && (
            <div style={{ color: COLORS.text.muted, fontSize: 14, textAlign: 'center', padding: '60px 20px' }}>
              Select a forecast report from the list to view its complete audit analysis.
            </div>
          )}

          {detailLoading && (
            <div style={{ color: COLORS.primary, padding: '40px 20px', textAlign: 'center', fontSize: 14 }}>
              Loading report details…
            </div>
          )}

          {detailError && (
            <div style={{ color: COLORS.accent, background: `${COLORS.accent}14`, padding: 14, borderRadius: 10, border: `1px solid ${COLORS.accent}33` }}>
              {detailError}
            </div>
          )}

          {detail && !detailLoading && (
            <div>
              {/* Report Header Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, borderBottom: `1px solid ${COLORS.bg.border}`, paddingBottom: 16, marginBottom: 16, flexWrap: 'wrap' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <h3 style={{ margin: 0, fontSize: 20, fontWeight: 900, textTransform: 'capitalize', color: COLORS.text.primary }}>
                      {String(detail.model || '').replace(/_/g, ' ')}
                    </h3>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 800,
                        padding: '3px 8px',
                        borderRadius: 5,
                        background: detail.domain === 'electricity' ? '#00D4FF22' : '#00E5A022',
                        color: detail.domain === 'electricity' ? COLORS.primary : COLORS.success,
                        textTransform: 'uppercase',
                      }}
                    >
                      {detail.domain}
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        padding: '3px 8px',
                        borderRadius: 5,
                        background: detail.forecast_type === 'future' ? '#FFB80022' : '#8899CC22',
                        color: detail.forecast_type === 'future' ? COLORS.warning : COLORS.text.secondary,
                      }}
                    >
                      {detail.forecast_type === 'future' ? 'Future Forecast' : 'Held-out Test Evaluation'}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: COLORS.text.secondary }}>
                    Run ID: <code style={{ color: COLORS.primary, background: `${COLORS.primary}11`, padding: '2px 6px', borderRadius: 4 }}>{detail.run_id}</code>
                    {' · '}
                    Executed: {detail.timestamp ? new Date(detail.timestamp).toLocaleString() : '—'}
                  </div>
                </div>

                {/* Export Action Toolbar */}
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    onClick={() => handleDownloadPdf(detail.run_id)}
                    disabled={actionLoading === 'pdf'}
                    style={{
                      background: COLORS.primary,
                      color: COLORS.bg.dark,
                      border: 'none',
                      borderRadius: 8,
                      padding: '7px 14px',
                      fontSize: 12,
                      fontWeight: 800,
                      cursor: actionLoading === 'pdf' ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                    }}
                  >
                    {actionLoading === 'pdf' ? 'Generating PDF…' : '↓ Download PDF Report'}
                  </button>

                  <button
                    type="button"
                    onClick={() => handleExportData(detail.run_id, 'csv')}
                    disabled={actionLoading === 'csv'}
                    style={{
                      background: 'transparent',
                      color: COLORS.text.primary,
                      border: `1px solid ${COLORS.bg.border}`,
                      borderRadius: 8,
                      padding: '7px 12px',
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: actionLoading === 'csv' ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Export CSV
                  </button>

                  <button
                    type="button"
                    onClick={() => handleExportData(detail.run_id, 'json')}
                    disabled={actionLoading === 'json'}
                    style={{
                      background: 'transparent',
                      color: COLORS.text.primary,
                      border: `1px solid ${COLORS.bg.border}`,
                      borderRadius: 8,
                      padding: '7px 12px',
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: actionLoading === 'json' ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Export JSON
                  </button>
                </div>
              </div>

              {actionMessage && (
                <div
                  style={{
                    marginBottom: 16,
                    padding: '8px 14px',
                    borderRadius: 8,
                    fontSize: 12,
                    fontWeight: 600,
                    background: actionMessage.type === 'success' ? `${COLORS.success}18` : `${COLORS.accent}18`,
                    border: `1px solid ${actionMessage.type === 'success' ? COLORS.success : COLORS.accent}44`,
                    color: actionMessage.type === 'success' ? COLORS.success : COLORS.accent,
                  }}
                >
                  {actionMessage.text}
                </div>
              )}

              {/* 1. Forecast Summary Card */}
              {detail.forecast_summary && (
                <div
                  style={{
                    marginBottom: 16,
                    padding: 14,
                    borderRadius: 10,
                    background: `${COLORS.primary}0a`,
                    border: `1px solid ${COLORS.primary}33`,
                  }}
                >
                  <div style={{ color: COLORS.primary, fontWeight: 800, fontSize: 11, letterSpacing: '0.06em', marginBottom: 6 }}>
                    EXECUTIVE FORECAST SUMMARY
                  </div>
                  <div style={{ color: COLORS.text.primary, fontSize: 13, lineHeight: 1.6 }}>{detail.forecast_summary.summary}</div>
                  {detail.forecast_summary.interpretation && (
                    <div style={{ marginTop: 6, color: COLORS.text.secondary, fontSize: 12, lineHeight: 1.5 }}>
                      {detail.forecast_summary.interpretation}
                    </div>
                  )}
                  {detail.forecast_summary.insights?.length > 0 && (
                    <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {detail.forecast_summary.insights.map((ins) => (
                        <div key={ins} style={{ display: 'flex', gap: 6, fontSize: 12, color: COLORS.text.secondary }}>
                          <span style={{ color: COLORS.primary }}>•</span>
                          <span>{ins}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* 2. Dataset & Context Grid */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em', marginBottom: 8 }}>
                  DATASET &amp; FORECASTING CONTEXT
                </div>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                    gap: 10,
                    background: COLORS.bg.innerBox,
                    padding: 12,
                    borderRadius: 10,
                    border: `1px solid ${COLORS.bg.border}`,
                  }}
                >
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.text.muted }}>Target Variable</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary }}>{detail.target || '—'}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.text.muted }}>Geographic Scope</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary }}>
                      {detail.dataset_info?.geographic_scope || 'All Geographies'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.text.muted }}>Observations / Horizon</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary }}>
                      {detail.forecast_type === 'future' ? `${detail.horizon ?? '—'} future steps` : `${detail.prediction_count ?? '—'} test rows`}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.text.muted }}>Frequency</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary }}>
                      {detail.dataset_info?.frequency || (detail.domain === 'electricity' ? 'Hourly (h)' : 'Daily (D)')}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.text.muted }}>Model Artifact</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, textTransform: 'capitalize' }}>
                      {detail.artifact_source || 'Trained'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.text.muted }}>Dataset Hash</div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, wordBreak: 'break-all' }}>
                      {detail.dataset_info?.dataset_hash?.substring(0, 12) || 'Canonical'}
                    </div>
                  </div>
                </div>
              </div>

              {/* 3. Performance Metrics Cards */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em' }}>
                    {detail.forecast_type === 'future' ? 'PROJECTED SCENARIO SUMMARY' : 'HELD-OUT TEST EVALUATION'}
                  </div>
                  <span style={{ fontSize: 10, color: COLORS.primary, fontWeight: 700 }}>
                    {detail.forecast_type === 'future' ? 'Future Projections' : 'Chronologically Unseen Test Split'}
                  </span>
                </div>

                {detail.forecast_type === 'future' ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                    {[
                      ['Projected Mean', formatNumber(detail.forecast_data?.summary?.mean)],
                      ['Projected Min', formatNumber(detail.forecast_data?.summary?.min)],
                      ['Projected Max', formatNumber(detail.forecast_data?.summary?.max)],
                      ['Forecast Direction', String(detail.summary_facts?.forecast_direction || 'Stable').toUpperCase()],
                    ].map(([label, val]) => (
                      <div
                        key={label}
                        style={{
                          background: COLORS.bg.innerBox,
                          border: `1px solid ${COLORS.bg.border}`,
                          borderRadius: 10,
                          padding: '12px 14px',
                        }}
                      >
                        <div style={{ fontSize: 11, color: COLORS.text.muted }}>{label}</div>
                        <div style={{ fontSize: 16, fontWeight: 800, color: COLORS.primary, marginTop: 4 }}>{val}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                    {[
                      ['Test RMSE', formatNumber(detail.metrics?.rmse)],
                      ['Test MAE', formatNumber(detail.metrics?.mae)],
                      ['Test MAPE', formatMape(detail.metrics?.mape)],
                      ['Test R²', formatNumber(detail.metrics?.r2, 3)],
                      ['Test WAPE', formatMape(detail.metrics?.wape)],
                    ].map(([label, val]) => (
                      <div
                        key={label}
                        style={{
                          background: COLORS.bg.innerBox,
                          border: `1px solid ${COLORS.bg.border}`,
                          borderRadius: 10,
                          padding: '12px 14px',
                        }}
                      >
                        <div style={{ fontSize: 11, color: COLORS.text.muted }}>{label}</div>
                        <div style={{ fontSize: 16, fontWeight: 800, color: COLORS.text.primary, marginTop: 4 }}>{val}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* 4. Model Selection & Validation Benchmark */}
              {detail.validation_benchmark?.ranking?.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div style={{ fontSize: 12, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em' }}>
                      VALIDATION BENCHMARK &amp; SELECTION RANKING
                    </div>
                    <span style={{ fontSize: 10, color: COLORS.success, fontWeight: 700 }}>
                      Selected on Validation Set (Lowest RMSE)
                    </span>
                  </div>

                  <div style={{ overflowX: 'auto', background: COLORS.bg.innerBox, borderRadius: 10, border: `1px solid ${COLORS.bg.border}` }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                      <thead>
                        <tr style={{ borderBottom: `1px solid ${COLORS.bg.border}`, color: COLORS.text.muted, textAlign: 'left' }}>
                          <th style={{ padding: '8px 12px' }}>Rank</th>
                          <th style={{ padding: '8px 12px' }}>Candidate Model</th>
                          <th style={{ padding: '8px 12px' }}>Val RMSE</th>
                          <th style={{ padding: '8px 12px' }}>Val MAE</th>
                          <th style={{ padding: '8px 12px' }}>Val MAPE</th>
                          <th style={{ padding: '8px 12px' }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.validation_benchmark.ranking.map((item, idx) => {
                          const mName = item.model || '';
                          const isWinner = idx === 0 || mName.toLowerCase() === detail.model?.toLowerCase();
                          const vMetrics = item.validation_metrics || item.metrics || {};
                          return (
                            <tr
                              key={mName || idx}
                              style={{
                                borderBottom: `1px solid ${COLORS.bg.border}55`,
                                background: isWinner ? `${COLORS.primary}10` : 'transparent',
                              }}
                            >
                              <td style={{ padding: '8px 12px', fontWeight: 700, color: isWinner ? COLORS.primary : COLORS.text.secondary }}>
                                #{idx + 1}
                              </td>
                              <td style={{ padding: '8px 12px', fontWeight: 700, color: isWinner ? COLORS.primary : COLORS.text.primary, textTransform: 'capitalize' }}>
                                {mName.replace(/_/g, ' ')}
                              </td>
                              <td style={{ padding: '8px 12px' }}>{formatNumber(vMetrics.rmse)}</td>
                              <td style={{ padding: '8px 12px' }}>{formatNumber(vMetrics.mae)}</td>
                              <td style={{ padding: '8px 12px' }}>{formatMape(vMetrics.mape)}</td>
                              <td style={{ padding: '8px 12px' }}>
                                {isWinner ? (
                                  <span style={{ color: COLORS.success, fontWeight: 800, fontSize: 11 }}>★ SELECTED</span>
                                ) : (
                                  <span style={{ color: COLORS.text.muted, fontSize: 11 }}>Candidate</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 5. Forecast Data Table (Sample Preview) */}
              {detail.forecast_data?.points?.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div style={{ fontSize: 12, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em' }}>
                      FORECAST TIME-SERIES OBSERVATIONS (SAMPLE)
                    </div>
                    <span style={{ fontSize: 10, color: COLORS.text.muted }}>
                      Showing {Math.min(15, detail.forecast_data.points.length)} of {detail.forecast_data.points.length} points
                    </span>
                  </div>

                  <div style={{ maxHeight: 220, overflowY: 'auto', background: COLORS.bg.innerBox, borderRadius: 10, border: `1px solid ${COLORS.bg.border}` }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                      <thead style={{ position: 'sticky', top: 0, background: COLORS.bg.cardHeader }}>
                        <tr style={{ borderBottom: `1px solid ${COLORS.bg.border}`, color: COLORS.text.muted, textAlign: 'left' }}>
                          <th style={{ padding: '6px 10px' }}>Timestamp</th>
                          <th style={{ padding: '6px 10px' }}>Predicted</th>
                          {detail.forecast_type === 'test_evaluation' && (
                            <>
                              <th style={{ padding: '6px 10px' }}>Actual</th>
                              <th style={{ padding: '6px 10px' }}>Residual</th>
                            </>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {detail.forecast_data.points.slice(0, 15).map((pt, i) => (
                          <tr key={pt.timestamp || i} style={{ borderBottom: `1px solid ${COLORS.bg.border}33` }}>
                            <td style={{ padding: '5px 10px', color: COLORS.text.secondary }}>
                              {String(pt.timestamp).replace('T', ' ').substring(0, 19)}
                            </td>
                            <td style={{ padding: '5px 10px', fontWeight: 700, color: COLORS.primary }}>
                              {formatNumber(pt.predicted)}
                            </td>
                            {detail.forecast_type === 'test_evaluation' && (
                              <>
                                <td style={{ padding: '5px 10px', color: COLORS.text.primary }}>{formatNumber(pt.actual)}</td>
                                <td
                                  style={{
                                    padding: '5px 10px',
                                    color: (pt.residual ?? 0) >= 0 ? COLORS.success : COLORS.accent,
                                  }}
                                >
                                  {formatNumber(pt.residual)}
                                </td>
                              </>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 6. Feature Attribution & Explainability */}
              {(detail.explainability?.feature_importance || detail.feature_importance) && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 12, fontWeight: 800, color: COLORS.text.muted, letterSpacing: '0.05em', marginBottom: 8 }}>
                    PREDICTIVE DRIVERS &amp; ATTRIBUTION (TOP FEATURES)
                  </div>
                  <div style={{ background: COLORS.bg.innerBox, borderRadius: 10, border: `1px solid ${COLORS.bg.border}`, padding: 12 }}>
                    {(() => {
                      const rawFeats = detail.explainability?.feature_importance || detail.feature_importance;
                      const entries = Object.entries(rawFeats || {}).sort((a, b) => Number(b[1]) - Number(a[1])).slice(0, 5);
                      const maxVal = entries[0]?.[1] || 1;
                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {entries.map(([fName, val]) => (
                            <div key={fName} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                              <span style={{ width: 140, fontSize: 11, fontWeight: 700, color: COLORS.text.secondary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {fName}
                              </span>
                              <div style={{ flex: 1, background: `${COLORS.primary}18`, borderRadius: 4, height: 8, overflow: 'hidden' }}>
                                <div style={{ background: COLORS.primary, width: `${(Number(val) / maxVal) * 100}%`, height: '100%', borderRadius: 4 }} />
                              </div>
                              <span style={{ width: 50, fontSize: 11, fontWeight: 800, color: COLORS.primary, textAlign: 'right' }}>
                                {formatNumber(val, 3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}

              {/* Backward compatibility check banner */}
              {'predictions' in detail && (
                <div style={{ marginTop: 10, color: COLORS.accent, fontSize: 12 }}>
                  Notice: Legacy prediction list detected in payload.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
