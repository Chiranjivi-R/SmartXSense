// SmartXSense: Explainable Hybrid Demand Forecasting Dashboard
// Main shell + navigation. Operational pages load live API data.

import { useState, useEffect } from "react";
import ForecastDashboard from "./components/forecasting/ForecastDashboard";
import OverviewDashboard from "./components/overview/OverviewDashboard";
import ExplainabilityPage from "./components/xai/ExplainabilityPage";
import ModelsCatalogPage from "./components/models/ModelsCatalogPage";
import ReportsPage from "./components/reports/ReportsPage";
import SystemSettingsPage from "./components/settings/SystemSettingsPage";

// ─── Color Palette ────────────────────────────────────────────────────────────
const COLORS = {
  primary: "#00D4FF",
  secondary: "#7B61FF",
  accent: "#FF6B6B",
  success: "#00E5A0",
  warning: "#FFB800",
  bg: {
    dark: "#060B1A",
    card: "#0D1530",
    cardHover: "#121B3A",
    border: "#1E2D5A",
  },
  text: {
    primary: "#E8F0FF",
    secondary: "#8899CC",
    muted: "#4A5B8A",
  },
};

const notifications = [
  { id: 1, type: "info", msg: "Demo tip: Retail → Random Forest → Forecast is the core evaluation path.", time: "Guide" },
  { id: 2, type: "info", msg: "Compare / Auto Select chooses models using validation RMSE only.", time: "Guide" },
  { id: 3, type: "info", msg: "Future forecasts never include actual values — predicted series only.", time: "Guide" },
  { id: 4, type: "info", msg: "History and exports store metadata/client downloads — not server filesystem paths.", time: "Guide" },
];

// ─── Shared Components ────────────────────────────────────────────────────────
const GlowBadge = ({ children, color = COLORS.primary }) => (
  <span style={{
    background: `${color}22`, color, border: `1px solid ${color}44`,
    borderRadius: 6, padding: "2px 10px", fontSize: 11, fontWeight: 700,
    letterSpacing: "0.05em", textTransform: "uppercase",
  }}>{children}</span>
);




// ─── Navigation Sidebar ───────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: "home", icon: "🏠", label: "Home" },
  { id: "dashboard", icon: "📊", label: "Dashboard" },
  { id: "retail", icon: "🛒", label: "Retail" },
  { id: "electricity", icon: "⚡", label: "Electricity" },
  { id: "forecast", icon: "📈", label: "Forecasts" },
  { id: "xai", icon: "🔍", label: "Explainability" },
  { id: "models", icon: "🤖", label: "AI Models" },
  { id: "reports", icon: "📄", label: "Reports" },
  { id: "settings", icon: "⚙️", label: "Settings" },
];

const Sidebar = ({ active, setActive, collapsed, setCollapsed }) => (
  <div style={{
    width: collapsed ? 64 : 220, minHeight: "100vh", background: COLORS.bg.card,
    borderRight: `1px solid ${COLORS.bg.border}`, display: "flex", flexDirection: "column",
    transition: "width 0.3s ease", position: "fixed", top: 0, left: 0, zIndex: 100, overflow: "hidden",
  }}>
    <div style={{ padding: collapsed ? "20px 16px" : "20px 20px", borderBottom: `1px solid ${COLORS.bg.border}`, display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ width: 34, height: 34, borderRadius: 10, background: `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.secondary})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>⚡</div>
      {!collapsed && <span style={{ fontWeight: 900, fontSize: 15, color: COLORS.text.primary, whiteSpace: "nowrap", fontFamily: "'Space Grotesk', sans-serif", letterSpacing: "-0.02em" }}>SmartXSense</span>}
    </div>
    <nav style={{ flex: 1, padding: "12px 8px" }}>
      {NAV_ITEMS.map(item => (
        <button key={item.id} onClick={() => setActive(item.id)} style={{
          width: "100%", display: "flex", alignItems: "center", gap: 12, padding: collapsed ? "10px 14px" : "10px 14px",
          borderRadius: 10, border: "none", cursor: "pointer", marginBottom: 2,
          background: active === item.id ? `linear-gradient(135deg, ${COLORS.primary}22, ${COLORS.secondary}22)` : "transparent",
          color: active === item.id ? COLORS.primary : COLORS.text.secondary,
          fontWeight: active === item.id ? 700 : 400, fontSize: 14, transition: "all 0.15s",
          borderLeft: active === item.id ? `3px solid ${COLORS.primary}` : "3px solid transparent",
          textAlign: "left", whiteSpace: "nowrap",
        }}>
          <span style={{ fontSize: 18, flexShrink: 0 }}>{item.icon}</span>
          {!collapsed && item.label}
        </button>
      ))}
    </nav>
    <button onClick={() => setCollapsed(!collapsed)} style={{
      margin: 8, border: `1px solid ${COLORS.bg.border}`, borderRadius: 10, background: "transparent",
      color: COLORS.text.secondary, cursor: "pointer", padding: "8px 12px", fontSize: 13,
    }}>
      {collapsed ? "→" : "← Collapse"}
    </button>
  </div>
);

// ─── Top Bar ──────────────────────────────────────────────────────────────────
const TopBar = ({ page, showNotifs, setShowNotifs }) => {
  const [search, setSearch] = useState("");
  return (
    <div style={{
      height: 62, background: COLORS.bg.card, borderBottom: `1px solid ${COLORS.bg.border}`,
      display: "flex", alignItems: "center", padding: "0 24px", gap: 16,
    }}>
      <div style={{ flex: 1, position: "relative" }}>
        <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: COLORS.text.muted }}>🔍</span>
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search forecasts, models, regions…"
          style={{
            width: "100%", maxWidth: 360, background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`,
            borderRadius: 10, padding: "8px 12px 8px 36px", color: COLORS.text.primary, fontSize: 14,
            outline: "none",
          }} />
      </div>
      <GlowBadge color={COLORS.success}>Local</GlowBadge>
      <div style={{ fontSize: 12, color: COLORS.text.muted, fontFamily: "'Space Mono', monospace" }}>
        {new Date().toLocaleString()}
      </div>
      <button onClick={() => setShowNotifs(!showNotifs)} style={{
        position: "relative", background: "transparent", border: `1px solid ${COLORS.bg.border}`,
        borderRadius: 10, padding: "7px 12px", color: COLORS.text.secondary, cursor: "pointer", fontSize: 16,
      }}>
        🔔
        <span style={{
          position: "absolute", top: 4, right: 4, width: 8, height: 8, borderRadius: "50%",
          background: COLORS.accent,
        }} />
      </button>
      <div style={{
        width: 36, height: 36, borderRadius: "50%",
        background: `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.secondary})`,
        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700,
        color: "#fff", cursor: "pointer",
      }}>A</div>
    </div>
  );
};

// ─── Notification Panel ───────────────────────────────────────────────────────
const NotifPanel = ({ onClose }) => (
  <div style={{
    position: "fixed", top: 70, right: 20, width: 360, background: COLORS.bg.card,
    border: `1px solid ${COLORS.bg.border}`, borderRadius: 16, zIndex: 200,
    boxShadow: `0 20px 60px #00000088`,
  }}>
    <div style={{ padding: "14px 18px", borderBottom: `1px solid ${COLORS.bg.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontWeight: 700, color: COLORS.text.primary, fontSize: 14 }}>Tips (static)</span>
      <button onClick={onClose} style={{ background: "transparent", border: "none", color: COLORS.text.muted, cursor: "pointer", fontSize: 18 }}>×</button>
    </div>
    {notifications.map(n => (
      <div key={n.id} style={{ padding: "12px 18px", borderBottom: `1px solid ${COLORS.bg.border}20`, display: "flex", gap: 12 }}>
        <span style={{ fontSize: 18 }}>
          {n.type === "alert" ? "🚨" : n.type === "success" ? "✅" : n.type === "warning" ? "⚠️" : "ℹ️"}
        </span>
        <div>
          <div style={{ fontSize: 13, color: COLORS.text.primary, lineHeight: 1.4 }}>{n.msg}</div>
          <div style={{ fontSize: 11, color: COLORS.text.muted, marginTop: 4 }}>{n.time}</div>
        </div>
      </div>
    ))}
  </div>
);

// ─── Landing / Home Page ──────────────────────────────────────────────────────
const HomePage = ({ setActive }) => {
  const features = [
    { icon: "🧠", title: "Hybrid AI Models", desc: "Tree ensembles, deep sequence models, and statistical baselines for demand forecasting" },
    { icon: "🔍", title: "Explainable AI", desc: "SHAP/LIME-based insights that reveal why each prediction was made" },
    { icon: "⚡", title: "End-to-End Forecasting", desc: "Upload datasets, evaluate on held-out tests, then run future horizons locally" },
    { icon: "🌡️", title: "Weather Integration", desc: "Exogenous variables including temperature, rainfall, and humidity" },
    { icon: "📊", title: "Interactive Analytics", desc: "Drill down into trends, anomalies, and seasonal patterns" },
    { icon: "🛡️", title: "Confidence Intervals", desc: "Probabilistic forecasts with uncertainty quantification" },
  ];

  return (
    <div style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
      {/* Hero */}
      <div style={{
        minHeight: "90vh", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center",
        textAlign: "center", padding: "60px 40px", position: "relative", overflow: "hidden",
      }}>
        {/* Animated background grid */}
        <div style={{
          position: "absolute", inset: 0, zIndex: 0,
          backgroundImage: `radial-gradient(${COLORS.primary}08 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }} />
        <div style={{
          position: "absolute", inset: 0, zIndex: 0,
          background: `radial-gradient(ellipse 60% 50% at 50% 40%, ${COLORS.secondary}18, transparent)`,
        }} />

        <div style={{ position: "relative", zIndex: 1, maxWidth: 760 }}>
          <GlowBadge>Next-Gen Forecasting Platform</GlowBadge>
          <h1 style={{
            fontSize: "clamp(32px, 5vw, 58px)", fontWeight: 900, margin: "24px 0 20px",
            color: COLORS.text.primary, lineHeight: 1.1, letterSpacing: "-0.03em",
          }}>
            <span style={{ color: COLORS.primary }}>SmartX</span>
            <span style={{ color: COLORS.secondary }}>Sense</span>
          </h1>
          <p style={{ fontSize: 20, color: COLORS.text.secondary, maxWidth: 560, margin: "0 auto 16px", lineHeight: 1.6 }}>
            Explainable Hybrid Demand Forecasting for Retail &amp; Electricity Systems
          </p>
          <p style={{ fontSize: 15, color: COLORS.text.muted, maxWidth: 520, margin: "0 auto 40px", lineHeight: 1.7 }}>
            Harness the power of interpretable AI to predict demand with precision — from store shelves to smart grids. Understand not just <em>what</em> will happen, but <em>why</em>.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <button onClick={() => setActive("dashboard")} style={{
              background: `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.secondary})`,
              border: "none", borderRadius: 12, padding: "14px 32px", color: "#fff",
              fontWeight: 700, fontSize: 15, cursor: "pointer", transition: "all 0.2s",
            }}
              onMouseEnter={e => { e.currentTarget.style.transform = "scale(1.04)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
            >
              Open Dashboard →
            </button>
            <button onClick={() => setActive("xai")} style={{
              background: "transparent", border: `1px solid ${COLORS.bg.border}`,
              borderRadius: 12, padding: "14px 32px", color: COLORS.text.primary,
              fontWeight: 600, fontSize: 15, cursor: "pointer", transition: "all 0.2s",
            }}>
              Explore Explainability
            </button>
          </div>
        </div>

        {/* Decorative capability chips (hidden on small layouts; marketing-safe labels only) */}
        {[
          { label: "Domains", val: "2", top: "15%", left: "5%", c: COLORS.success },
          { label: "Model Families", val: "8", top: "25%", right: "4%", c: COLORS.primary },
          { label: "XAI", val: "SHAP/LIME", bottom: "20%", left: "3%", c: COLORS.secondary },
          { label: "Storage", val: "File Artifacts", bottom: "30%", right: "3%", c: COLORS.warning },
        ].map((b, i) => (
          <div key={i} style={{
            position: "absolute", ...b, background: COLORS.bg.card,
            border: `1px solid ${b.c}44`, borderRadius: 14, padding: "10px 16px",
            textAlign: "center", zIndex: 1, display: "none",
          }}>
            <div style={{ fontSize: 11, color: COLORS.text.muted, textTransform: "uppercase", letterSpacing: "0.06em" }}>{b.label}</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: b.c, fontFamily: "'Space Mono', monospace" }}>{b.val}</div>
          </div>
        ))}
      </div>

      {/* Product capability bar — not live telemetry */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: 1, background: COLORS.bg.border, borderTop: `1px solid ${COLORS.bg.border}`, borderBottom: `1px solid ${COLORS.bg.border}`,
      }}>
        {[
          { label: "Domains", val: "Retail + Electricity" },
          { label: "Core Models", val: "RF · XGB · ARIMA · Hybrid" },
          { label: "Explainability", val: "SHAP + LIME" },
          { label: "Workflow", val: "Upload → Forecast → Export" },
        ].map((s, i) => (
          <div key={i} style={{ background: COLORS.bg.dark, padding: "22px 24px", textAlign: "center" }}>
            <div style={{ fontSize: 18, fontWeight: 900, color: COLORS.primary, fontFamily: "'Space Mono', monospace" }}>{s.val}</div>
            <div style={{ fontSize: 12, color: COLORS.text.muted, marginTop: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Features */}
      <div style={{ padding: "60px 40px" }}>
        <div style={{ textAlign: "center", marginBottom: 48 }}>
          <h2 style={{ fontSize: 32, fontWeight: 900, color: COLORS.text.primary, margin: "0 0 12px", letterSpacing: "-0.03em" }}>Platform Capabilities</h2>
          <p style={{ color: COLORS.text.secondary, fontSize: 16 }}>Everything you need for intelligent demand planning</p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20, maxWidth: 1000, margin: "0 auto" }}>
          {features.map((f, i) => (
            <div key={i} style={{
              background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`, borderRadius: 16,
              padding: "24px 24px", transition: "all 0.2s", cursor: "default",
            }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = `${COLORS.primary}66`; e.currentTarget.style.transform = "translateY(-3px)"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = COLORS.bg.border; e.currentTarget.style.transform = "translateY(0)"; }}
            >
              <div style={{ fontSize: 32, marginBottom: 14 }}>{f.icon}</div>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary, margin: "0 0 8px" }}>{f.title}</h3>
              <p style={{ fontSize: 14, color: COLORS.text.secondary, margin: 0, lineHeight: 1.6 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Benefits of XAI */}
      <div style={{ background: COLORS.bg.card, padding: "60px 40px", borderTop: `1px solid ${COLORS.bg.border}` }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <h2 style={{ fontSize: 28, fontWeight: 900, color: COLORS.text.primary, textAlign: "center", margin: "0 0 12px" }}>Why Explainable AI?</h2>
          <p style={{ color: COLORS.text.secondary, textAlign: "center", fontSize: 15, marginBottom: 40 }}>Black boxes breed distrust. SmartXSense shows its work.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 24 }}>
            {[
              { title: "Regulatory Compliance", desc: "Meet GDPR, EU AI Act, and sector-specific requirements for model transparency", color: COLORS.primary },
              { title: "Stakeholder Trust", desc: "Operations and finance teams act confidently when they understand the rationale", color: COLORS.secondary },
              { title: "Faster Debugging", desc: "Quickly identify data drift, spurious correlations, or model degradation", color: COLORS.success },
              { title: "Domain Integration", desc: "Human experts validate and override predictions with context the model lacks", color: COLORS.warning },
            ].map((b, i) => (
              <div key={i} style={{ borderLeft: `3px solid ${b.color}`, paddingLeft: 20 }}>
                <h4 style={{ color: b.color, fontSize: 15, fontWeight: 700, margin: "0 0 8px" }}>{b.title}</h4>
                <p style={{ color: COLORS.text.secondary, fontSize: 13, margin: 0, lineHeight: 1.6 }}>{b.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [active, setActive] = useState("home");
  const [collapsed, setCollapsed] = useState(false);
  const [showNotifs, setShowNotifs] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Loading animation
    const t = setTimeout(() => setLoading(false), 1200);
    return () => clearTimeout(t);
  }, []);

  // Loading screen
  if (loading) return (
    <div style={{
      minHeight: "100vh", background: COLORS.bg.dark, display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", fontFamily: "'Space Grotesk', sans-serif",
    }}>
      <div style={{ fontSize: 48, marginBottom: 24, animation: "spin 2s linear infinite" }}>⚡</div>
      <div style={{ fontSize: 28, fontWeight: 900, color: COLORS.text.primary }}>
        <span style={{ color: COLORS.primary }}>SmartX</span><span style={{ color: COLORS.secondary }}>Sense</span>
      </div>
      <div style={{ fontSize: 14, color: COLORS.text.muted, marginTop: 8 }}>Loading forecasting platform…</div>
      <div style={{ marginTop: 32, width: 200, height: 3, background: COLORS.bg.border, borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 3,
          background: `linear-gradient(90deg, ${COLORS.primary}, ${COLORS.secondary})`,
          animation: "load 1.2s ease-out forwards",
        }} />
      </div>
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes load { from { width: 0%; } to { width: 100%; } }
      `}</style>
    </div>
  );

  const sideW = collapsed ? 64 : 220;
  const isHome = active === "home";

  const pageComponents = {
    home: <HomePage setActive={setActive} />,
    dashboard: <OverviewDashboard onNavigate={setActive} />,
    retail: <ForecastDashboard initialDomain="retail" />,
    electricity: <ForecastDashboard initialDomain="electricity" />,
    forecast: <ForecastDashboard initialDomain="retail" />,
    xai: <ExplainabilityPage />,
    models: <ModelsCatalogPage />,
    reports: <ReportsPage />,
    settings: <SystemSettingsPage />,
  };

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg.dark, fontFamily: "'Space Grotesk', sans-serif", color: COLORS.text.primary }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;900&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${COLORS.bg.dark}; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.bg.border}; border-radius: 3px; }
        input[type=range] { cursor: pointer; }
        select option { background: ${COLORS.bg.card}; }
      `}</style>

      {/* Sidebar */}
      <Sidebar active={active} setActive={setActive} collapsed={collapsed} setCollapsed={setCollapsed} />

      {/* Main Content */}
      <div style={{ marginLeft: sideW, transition: "margin-left 0.3s ease", minHeight: "100vh" }}>
        {/* TopBar always visible except home hero */}
        <div style={{ position: "sticky", top: 0, zIndex: 50 }}>
          <TopBar page={active} showNotifs={showNotifs} setShowNotifs={setShowNotifs} />
        </div>

        {/* Notification Panel */}
        {showNotifs && <NotifPanel onClose={() => setShowNotifs(false)} />}

        {/* Page Content */}
        <div style={{ padding: isHome ? 0 : "32px 32px" }}>
          {pageComponents[active] || pageComponents.dashboard}
        </div>
      </div>
    </div>
  );
}
