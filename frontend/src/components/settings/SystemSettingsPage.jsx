import React, { useEffect, useState } from 'react';
import { fetchSystemSettings } from '../../services/mlApi';

const COLORS = {
  primary: '#00D4FF',
  accent: '#FF6B6B',
  success: '#00E5A0',
  bg: { dark: '#060B1A', card: '#0D1530', border: '#1E2D5A' },
  text: { primary: '#E8F0FF', secondary: '#8899CC', muted: '#4A5B8A' },
};

export default function SystemSettingsPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await fetchSystemSettings());
    } catch (err) {
      setError(err.message || 'Unable to load system information.');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div style={{ maxWidth: 760 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, color: COLORS.text.primary }}>Settings</h2>
          <p style={{ margin: '6px 0 0', fontSize: 13, color: COLORS.text.secondary }}>
            Read-only system information from the live application.
          </p>
        </div>
        <button type="button" onClick={load} disabled={loading} style={refreshStyle(loading)}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ background: `${COLORS.accent}18`, border: `1px solid ${COLORS.accent}44`, borderRadius: 12, padding: 14, color: COLORS.accent, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {loading && !data && <div style={{ color: COLORS.text.secondary }}>Loading system information…</div>}

      {data && (
        <>
          <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>System Information</div>
            {[
              ['Name', data.application],
              ['Frontend Version', data.frontend_version],
              ['API Version', data.api_version],
              ['Backend Status', data.backend_status],
            ].map(([label, value]) => (
              <Row key={label} label={label} value={value} />
            ))}
          </div>

          <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>Application Capabilities</div>
            <Row label="Domains" value={(data.supported_domains || []).join(', ')} />
            <Row label="Models" value={(data.supported_models || []).join(', ')} />
          </div>

          <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>API Availability</div>
            {Object.entries(data.apis || {}).map(([key, value]) => (
              <Row key={key} label={key.replace(/_/g, ' ')} value={value ? 'Available' : 'Unavailable'} ok={Boolean(value)} />
            ))}
          </div>

          <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 14, padding: 18 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>Capability Flags</div>
            {Object.entries(data.capabilities || {}).map(([key, value]) => (
              <Row key={key} label={key.replace(/_/g, ' ')} value={value ? 'Enabled' : 'Disabled'} ok={Boolean(value)} />
            ))}
            {(data.notes || []).map((note) => (
              <div key={note} style={{ marginTop: 10, fontSize: 12, color: COLORS.text.muted }}>{note}</div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function Row({ label, value, ok }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, padding: '8px 0', borderBottom: '1px solid #1E2D5A', fontSize: 13 }}>
      <span style={{ color: '#4A5B8A', textTransform: 'capitalize' }}>{label}</span>
      <span style={{ color: ok === false ? '#FF6B6B' : ok ? '#00E5A0' : '#E8F0FF', textAlign: 'right' }}>{String(value)}</span>
    </div>
  );
}

const refreshStyle = (disabled) => ({
  background: '#060B1A',
  border: '1px solid #1E2D5A',
  color: '#00D4FF',
  borderRadius: 10,
  padding: '8px 14px',
  fontWeight: 700,
  cursor: disabled ? 'not-allowed' : 'pointer',
});
