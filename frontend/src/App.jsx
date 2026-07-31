// SmartXSense: Explainable Hybrid Demand Forecasting Dashboard
// Frontend-only React application with dummy data

import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

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

const CHART_COLORS = ["#00D4FF", "#7B61FF", "#FF6B6B", "#00E5A0", "#FFB800", "#FF9F43"];

// ─── Dummy Data ────────────────────────────────────────────────────────────────
const retailDailyData = [
  { day: "Mon", actual: 4200, predicted: 4100, lower: 3900, upper: 4300 },
  { day: "Tue", actual: 3800, predicted: 3900, lower: 3700, upper: 4100 },
  { day: "Wed", actual: 5100, predicted: 4950, lower: 4750, upper: 5150 },
  { day: "Thu", actual: 4700, predicted: 4800, lower: 4600, upper: 5000 },
  { day: "Fri", actual: 6200, predicted: 6100, lower: 5900, upper: 6300 },
  { day: "Sat", actual: 7800, predicted: 7650, lower: 7400, upper: 7900 },
  { day: "Sun", actual: 6500, predicted: 6600, lower: 6400, upper: 6800 },
];

const electricityData = [
  { hour: "00:00", consumption: 2100, predicted: 2050, baseline: 1800 },
  { hour: "03:00", consumption: 1800, predicted: 1850, baseline: 1600 },
  { hour: "06:00", consumption: 3200, predicted: 3100, baseline: 2800 },
  { hour: "09:00", consumption: 5400, predicted: 5300, baseline: 4800 },
  { hour: "12:00", consumption: 6800, predicted: 6750, baseline: 5900 },
  { hour: "15:00", consumption: 7200, predicted: 7100, baseline: 6200 },
  { hour: "18:00", consumption: 8100, predicted: 8000, baseline: 7000 },
  { hour: "21:00", consumption: 5600, predicted: 5500, baseline: 4900 },
];

const monthlyRetail = [
  { month: "Jan", sales: 42000, inventory: 38000, forecast: 44000 },
  { month: "Feb", sales: 38000, inventory: 35000, forecast: 40000 },
  { month: "Mar", sales: 51000, inventory: 47000, forecast: 52000 },
  { month: "Apr", sales: 47000, inventory: 43000, forecast: 48500 },
  { month: "May", sales: 62000, inventory: 55000, forecast: 63000 },
  { month: "Jun", sales: 78000, inventory: 70000, forecast: 79000 },
  { month: "Jul", sales: 65000, inventory: 59000, forecast: 66500 },
  { month: "Aug", sales: 72000, inventory: 65000, forecast: 73000 },
  { month: "Sep", sales: 58000, inventory: 53000, forecast: 59500 },
  { month: "Oct", sales: 69000, inventory: 62000, forecast: 70000 },
  { month: "Nov", sales: 91000, inventory: 82000, forecast: 92000 },
  { month: "Dec", sales: 115000, inventory: 103000, forecast: 116000 },
];

const categoryData = [
  { name: "Electronics", value: 35, growth: 12.4 },
  { name: "Apparel", value: 22, growth: 8.1 },
  { name: "Groceries", value: 18, growth: 5.3 },
  { name: "Home & Garden", value: 14, growth: 15.7 },
  { name: "Sports", value: 11, growth: 9.8 },
];

const featureImportance = [
  { feature: "Day of Week", importance: 0.24, direction: "positive" },
  { feature: "Temperature", importance: 0.18, direction: "positive" },
  { feature: "Holiday Flag", importance: 0.16, direction: "positive" },
  { feature: "Prev Week Sales", importance: 0.14, direction: "positive" },
  { feature: "Promotion Active", importance: 0.13, direction: "negative" },
  { feature: "Competitor Price", importance: 0.09, direction: "negative" },
  { feature: "Rainfall", importance: 0.06, direction: "negative" },
];

const modelComparison = [
  { name: "ARIMA", mape: 8.4, rmse: 342, accuracy: 91.6, latency: 12 },
  { name: "LSTM", mape: 5.2, rmse: 218, accuracy: 94.8, latency: 45 },
  { name: "XGBoost", mape: 4.8, rmse: 196, accuracy: 95.2, latency: 8 },
  { name: "Hybrid (Ours)", mape: 2.9, rmse: 124, accuracy: 97.1, latency: 18 },
];

const regionElectricity = [
  { region: "North", peak: 8200, avg: 5400, offpeak: 2100 },
  { region: "South", peak: 9100, avg: 6200, offpeak: 2800 },
  { region: "East", peak: 7400, avg: 4900, offpeak: 1900 },
  { region: "West", peak: 10200, avg: 7100, offpeak: 3200 },
  { region: "Central", peak: 6800, avg: 4500, offpeak: 1700 },
];

const radarData = [
  { subject: "Accuracy", ARIMA: 65, LSTM: 80, Hybrid: 95 },
  { subject: "Speed", ARIMA: 90, LSTM: 55, Hybrid: 75 },
  { subject: "Explainability", ARIMA: 85, LSTM: 40, Hybrid: 90 },
  { subject: "Scalability", ARIMA: 60, LSTM: 75, Hybrid: 92 },
  { subject: "Robustness", ARIMA: 70, LSTM: 72, Hybrid: 88 },
];

const notifications = [
  { id: 1, type: "alert", msg: "Peak electricity demand forecasted for 18:00–20:00 today", time: "2m ago" },
  { id: 2, type: "success", msg: "Retail forecast accuracy improved to 97.1% this week", time: "15m ago" },
  { id: 3, type: "warning", msg: "Inventory low for Electronics category — reorder suggested", time: "1h ago" },
  { id: 4, type: "info", msg: "Model retrained with last 30 days of data", time: "3h ago" },
];

// ─── Shared Components ────────────────────────────────────────────────────────
const GlowBadge = ({ children, color = COLORS.primary }) => (
  <span style={{
    background: `${color}22`, color, border: `1px solid ${color}44`,
    borderRadius: 6, padding: "2px 10px", fontSize: 11, fontWeight: 700,
    letterSpacing: "0.05em", textTransform: "uppercase",
  }}>{children}</span>
);

const StatCard = ({ icon, label, value, sub, delta, color = COLORS.primary }) => (
  <div style={{
    background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`,
    borderRadius: 14, padding: "18px 20px", position: "relative", overflow: "hidden",
    transition: "transform 0.2s, box-shadow 0.2s",
  }}
    onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = `0 8px 32px ${color}22`; }}
    onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "none"; }}
  >
    <div style={{ position: "absolute", right: -10, top: -10, fontSize: 64, opacity: 0.06 }}>{icon}</div>
    <div style={{ fontSize: 12, color: COLORS.text.secondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</div>
    <div style={{ fontSize: 26, fontWeight: 800, color: COLORS.text.primary, fontFamily: "'Space Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 12, color: COLORS.text.muted, marginTop: 4 }}>{sub}</div>}
    {delta !== undefined && (
      <div style={{ marginTop: 8, fontSize: 13, color: delta >= 0 ? COLORS.success : COLORS.accent, fontWeight: 600 }}>
        {delta >= 0 ? "▲" : "▼"} {Math.abs(delta)}% vs last week
      </div>
    )}
  </div>
);

const SectionTitle = ({ children, sub }) => (
  <div style={{ marginBottom: 24 }}>
    <h2 style={{ color: COLORS.text.primary, fontSize: 20, fontWeight: 800, margin: 0, fontFamily: "'Space Grotesk', sans-serif" }}>{children}</h2>
    {sub && <p style={{ color: COLORS.text.secondary, fontSize: 13, margin: "4px 0 0", fontFamily: "sans-serif" }}>{sub}</p>}
  </div>
);

const Card = ({ children, style = {} }) => (
  <div style={{
    background: COLORS.bg.card, border: `1px solid ${COLORS.bg.border}`,
    borderRadius: 16, padding: 24, ...style
  }}>{children}</div>
);

const customTooltipStyle = {
  backgroundColor: "#0D1530", border: "1px solid #1E2D5A",
  borderRadius: 10, color: "#E8F0FF", fontSize: 13,
};

// ─── Navigation Sidebar ───────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: "home", icon: "🏠", label: "Home" },
  { id: "dashboard", icon: "📊", label: "Dashboard" },
  { id: "retail", icon: "🛒", label: "Retail" },
  { id: "electricity", icon: "⚡", label: "Electricity" },
  { id: "xai", icon: "🔍", label: "Explainability" },
  { id: "forecast", icon: "📈", label: "Forecasts" },
  { id: "models", icon: "🤖", label: "AI Models" },
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
      <GlowBadge color={COLORS.success}>Live</GlowBadge>
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
      <span style={{ fontWeight: 700, color: COLORS.text.primary, fontSize: 14 }}>Notifications</span>
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
    { icon: "🧠", title: "Hybrid AI Models", desc: "Combines LSTM, XGBoost, and statistical models for superior accuracy" },
    { icon: "🔍", title: "Explainable AI", desc: "SHAP/LIME-based insights that reveal why each prediction was made" },
    { icon: "⚡", title: "Real-time Forecasting", desc: "Live demand signals processed and predicted within milliseconds" },
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

        {/* Floating metric bubbles */}
        {[
          { label: "MAPE", val: "2.9%", top: "15%", left: "5%", c: COLORS.success },
          { label: "Accuracy", val: "97.1%", top: "25%", right: "4%", c: COLORS.primary },
          { label: "Models", val: "4 Hybrid", bottom: "20%", left: "3%", c: COLORS.secondary },
          { label: "Uptime", val: "99.8%", bottom: "30%", right: "3%", c: COLORS.warning },
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

      {/* Stats bar */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: 1, background: COLORS.bg.border, borderTop: `1px solid ${COLORS.bg.border}`, borderBottom: `1px solid ${COLORS.bg.border}`,
      }}>
        {[
          { label: "Forecast Accuracy", val: "97.1%" },
          { label: "Data Points / Day", val: "2.4M+" },
          { label: "Sectors Covered", val: "Retail + Energy" },
          { label: "Latency", val: "<20ms" },
        ].map((s, i) => (
          <div key={i} style={{ background: COLORS.bg.dark, padding: "22px 24px", textAlign: "center" }}>
            <div style={{ fontSize: 26, fontWeight: 900, color: COLORS.primary, fontFamily: "'Space Mono', monospace" }}>{s.val}</div>
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

// ─── Dashboard Page ───────────────────────────────────────────────────────────
const DashboardPage = () => {
  const [period, setPeriod] = useState("weekly");

  return (
    <div>
      <SectionTitle sub="Real-time overview of all forecasting systems">Overview Dashboard</SectionTitle>

      {/* Stats Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 28 }}>
        <StatCard icon="🛒" label="Retail Forecast Accuracy" value="97.1%" delta={1.3} color={COLORS.primary} />
        <StatCard icon="⚡" label="Energy Forecast MAPE" value="2.9%" sub="Mean Absolute % Error" delta={-0.7} color={COLORS.success} />
        <StatCard icon="📦" label="SKUs Tracked" value="14,280" sub="Across 8 categories" color={COLORS.secondary} />
        <StatCard icon="🏙️" label="Grid Regions" value="5" sub="Peak: West at 10.2 GW" color={COLORS.warning} />
      </div>

      {/* Period Tabs */}
      <div style={{ display: "flex", gap: 6, marginBottom: 20 }}>
        {["daily", "weekly", "monthly"].map(p => (
          <button key={p} onClick={() => setPeriod(p)} style={{
            padding: "6px 18px", borderRadius: 8, border: `1px solid ${period === p ? COLORS.primary : COLORS.bg.border}`,
            background: period === p ? `${COLORS.primary}22` : "transparent",
            color: period === p ? COLORS.primary : COLORS.text.secondary,
            cursor: "pointer", fontSize: 13, fontWeight: 600, textTransform: "capitalize",
          }}>{p}</button>
        ))}
      </div>

      {/* Retail Forecast Chart */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, marginBottom: 20 }}>
        <Card>
          <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>
            Retail Demand Forecast — {period.charAt(0).toUpperCase() + period.slice(1)}
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={period === "monthly" ? monthlyRetail : retailDailyData}>
              <defs>
                <linearGradient id="colorAct" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.secondary} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS.secondary} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
              <XAxis dataKey={period === "monthly" ? "month" : "day"} tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
              <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
              <Tooltip contentStyle={customTooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
              <Area type="monotone" dataKey="actual" stroke={COLORS.primary} fill="url(#colorAct)" strokeWidth={2} name="Actual" dot={false} />
              <Area type="monotone" dataKey={period === "monthly" ? "forecast" : "predicted"} stroke={COLORS.secondary} fill="url(#colorPred)" strokeWidth={2} strokeDasharray="5 5" name="Predicted" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Category Share</div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={categoryData} cx="50%" cy="50%" innerRadius={50} outerRadius={85} paddingAngle={3} dataKey="value">
                {categoryData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={customTooltipStyle} formatter={(v) => `${v}%`} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
            {categoryData.map((c, i) => (
              <span key={i} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: COLORS.text.secondary }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: CHART_COLORS[i], flexShrink: 0 }} />
                {c.name}
              </span>
            ))}
          </div>
        </Card>
      </div>

      {/* Electricity & Region Charts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <Card>
          <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Electricity Consumption Today</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={electricityData}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
              <XAxis dataKey="hour" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
              <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
              <Tooltip contentStyle={customTooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
              <Line dataKey="consumption" stroke={COLORS.warning} strokeWidth={2} dot={false} name="Actual" />
              <Line dataKey="predicted" stroke={COLORS.primary} strokeWidth={2} strokeDasharray="5 5" dot={false} name="Predicted" />
              <Line dataKey="baseline" stroke={COLORS.text.muted} strokeWidth={1} dot={false} name="Baseline" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Region-wise Electricity Demand (MW)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={regionElectricity}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
              <XAxis dataKey="region" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
              <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
              <Tooltip contentStyle={customTooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
              <Bar dataKey="peak" fill={COLORS.accent} name="Peak" radius={[3, 3, 0, 0]} />
              <Bar dataKey="avg" fill={COLORS.primary} name="Average" radius={[3, 3, 0, 0]} />
              <Bar dataKey="offpeak" fill={COLORS.secondary} name="Off-Peak" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
};

// ─── Retail Forecasting Page ──────────────────────────────────────────────────
const RetailPage = () => (
  <div>
    <SectionTitle sub="Product demand trends, inventory planning, and sales analytics">Retail Forecasting</SectionTitle>

    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 28 }}>
      <StatCard icon="💰" label="Monthly Revenue" value="₹91.2L" delta={4.8} color={COLORS.primary} />
      <StatCard icon="📦" label="Active SKUs" value="14,280" color={COLORS.success} />
      <StatCard icon="🔄" label="Inventory Turns" value="8.3×" delta={1.1} color={COLORS.secondary} />
      <StatCard icon="🎯" label="Forecast Accuracy" value="97.1%" delta={1.3} color={COLORS.warning} />
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, marginBottom: 20 }}>
      <Card>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Monthly Sales vs Inventory vs Forecast</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={monthlyRetail}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
            <XAxis dataKey="month" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
            <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} tickFormatter={v => `${(v / 1000).toFixed(0)}k`} />
            <Tooltip contentStyle={customTooltipStyle} formatter={v => `₹${(v / 1000).toFixed(1)}k`} />
            <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
            <Bar dataKey="sales" fill={COLORS.primary} name="Sales" radius={[3, 3, 0, 0]} />
            <Bar dataKey="inventory" fill={COLORS.secondary} name="Inventory" radius={[3, 3, 0, 0]} />
            <Bar dataKey="forecast" fill={COLORS.success} name="Forecast" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginBottom: 12 }}>Category Growth Rates</div>
          {categoryData.map((c, i) => (
            <div key={i} style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 12, color: COLORS.text.secondary }}>{c.name}</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.success }}>+{c.growth}%</span>
              </div>
              <div style={{ background: COLORS.bg.dark, borderRadius: 4, height: 5 }}>
                <div style={{ width: `${c.growth * 5}%`, height: "100%", borderRadius: 4, background: CHART_COLORS[i] }} />
              </div>
            </div>
          ))}
        </Card>

        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginBottom: 12 }}>Reorder Alerts</div>
          {["Electronics — Low Stock", "Apparel — Reorder Now", "Sports — On Track"].map((a, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
              <span style={{ fontSize: 14 }}>{i === 0 ? "🟡" : i === 1 ? "🔴" : "🟢"}</span>
              <span style={{ fontSize: 13, color: COLORS.text.secondary }}>{a}</span>
            </div>
          ))}
          <button style={{
            marginTop: 8, width: "100%", padding: "8px 16px", borderRadius: 8,
            background: `${COLORS.primary}22`, border: `1px solid ${COLORS.primary}44`,
            color: COLORS.primary, fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}>Download Report ↓</button>
        </Card>
      </div>
    </div>

    {/* Customer Behavior */}
    <Card>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Daily Demand — Actual vs Predicted</div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={retailDailyData}>
          <defs>
            <linearGradient id="actGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.25} />
              <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
          <XAxis dataKey="day" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
          <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
          <Tooltip contentStyle={customTooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
          <Area type="monotone" dataKey="actual" stroke={COLORS.primary} fill="url(#actGrad)" strokeWidth={2} name="Actual" />
          <Line type="monotone" dataKey="predicted" stroke={COLORS.accent} strokeWidth={2} strokeDasharray="6 3" name="Predicted" dot={false} />
          <Line type="monotone" dataKey="upper" stroke={COLORS.text.muted} strokeWidth={1} strokeDasharray="3 3" name="Upper CI" dot={false} />
          <Line type="monotone" dataKey="lower" stroke={COLORS.text.muted} strokeWidth={1} strokeDasharray="3 3" name="Lower CI" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  </div>
);

// ─── Electricity Forecasting Page ─────────────────────────────────────────────
const ElectricityPage = () => (
  <div>
    <SectionTitle sub="Power consumption trends, peak analysis, and smart grid visualization">Electricity Forecasting</SectionTitle>

    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 28 }}>
      <StatCard icon="⚡" label="Current Load" value="7,840 MW" delta={3.2} color={COLORS.warning} />
      <StatCard icon="🔝" label="Peak Today" value="10.2 GW" sub="West Region" color={COLORS.accent} />
      <StatCard icon="🌿" label="Renewable Share" value="38.4%" delta={5.1} color={COLORS.success} />
      <StatCard icon="🎯" label="Grid MAPE" value="2.9%" delta={-0.4} color={COLORS.primary} />
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 20, marginBottom: 20 }}>
      <Card>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>24-Hour Consumption Forecast</div>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={electricityData}>
            <defs>
              <linearGradient id="elecGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={COLORS.warning} stopOpacity={0.35} />
                <stop offset="95%" stopColor={COLORS.warning} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
            <XAxis dataKey="hour" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
            <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
            <Tooltip contentStyle={customTooltipStyle} formatter={v => `${v.toLocaleString()} MW`} />
            <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
            <Area type="monotone" dataKey="consumption" stroke={COLORS.warning} fill="url(#elecGrad)" strokeWidth={2} name="Consumption (MW)" />
            <Line type="monotone" dataKey="predicted" stroke={COLORS.primary} strokeWidth={2} strokeDasharray="5 5" name="Predicted" dot={false} />
            <Line type="monotone" dataKey="baseline" stroke={COLORS.text.muted} strokeWidth={1} name="Baseline" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Peak vs Off-Peak by Region</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={regionElectricity} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
            <XAxis type="number" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
            <YAxis dataKey="region" type="category" tick={{ fill: COLORS.text.muted, fontSize: 11 }} width={55} />
            <Tooltip contentStyle={customTooltipStyle} formatter={v => `${v.toLocaleString()} MW`} />
            <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
            <Bar dataKey="peak" fill={COLORS.accent} name="Peak" radius={[0, 4, 4, 0]} />
            <Bar dataKey="avg" fill={COLORS.warning} name="Average" radius={[0, 4, 4, 0]} />
            <Bar dataKey="offpeak" fill={COLORS.secondary} name="Off-Peak" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>

    {/* Smart Grid Status */}
    <Card>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 18 }}>Smart Grid Status Panel</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14 }}>
        {[
          { region: "North", load: 5400, cap: 7000, status: "Normal" },
          { region: "South", load: 6200, cap: 7500, status: "Normal" },
          { region: "East", load: 4900, cap: 6000, status: "Normal" },
          { region: "West", load: 10200, cap: 11000, status: "⚠️ High" },
          { region: "Central", load: 4500, cap: 6500, status: "Normal" },
        ].map((r, i) => {
          const pct = Math.round((r.load / r.cap) * 100);
          const color = pct > 90 ? COLORS.accent : pct > 75 ? COLORS.warning : COLORS.success;
          return (
            <div key={i} style={{ background: COLORS.bg.dark, borderRadius: 12, padding: "16px 16px", border: `1px solid ${color}33` }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.text.primary, marginBottom: 10 }}>{r.region} Grid</div>
              <div style={{ fontSize: 20, fontWeight: 900, color, fontFamily: "'Space Mono', monospace", marginBottom: 6 }}>{pct}%</div>
              <div style={{ background: COLORS.bg.card, borderRadius: 4, height: 6, marginBottom: 8 }}>
                <div style={{ width: `${pct}%`, height: "100%", borderRadius: 4, background: color, transition: "width 1s" }} />
              </div>
              <div style={{ fontSize: 11, color: COLORS.text.muted }}>{r.load.toLocaleString()} / {r.cap.toLocaleString()} MW</div>
              <div style={{ fontSize: 11, color, marginTop: 4, fontWeight: 600 }}>{r.status}</div>
            </div>
          );
        })}
      </div>
    </Card>
  </div>
);

// ─── XAI (Explainability) Page ────────────────────────────────────────────────
const XAIPage = () => {
  const [selected, setSelected] = useState(null);

  const explanationCards = [
    {
      id: 1, prediction: "Saturday Sales: ₹78,000", confidence: 94,
      top_factors: ["Day=Saturday (+28%)", "Weather Sunny (+12%)", "Promotion Active (+9%)"],
      rationale: "Weekend combined with clear weather and an active promotional campaign drives consumer footfall significantly above the weekly mean.",
      color: COLORS.primary,
    },
    {
      id: 2, prediction: "Evening Peak: 8,100 MW", confidence: 91,
      top_factors: ["Hour=18:00 (+35%)", "Temperature=38°C (+18%)", "Working Day (+10%)"],
      rationale: "Post-work AC usage spikes sharply at 18:00 on hot weekdays. This pattern is highly consistent across all five regions.",
      color: COLORS.warning,
    },
    {
      id: 3, prediction: "Electronics Low Demand", confidence: 87,
      top_factors: ["Post-Payday Lag (−14%)", "Competitor Sale (−11%)", "Month=Mid (−8%)"],
      rationale: "Mid-month demand for discretionary electronics falls when competing retailers run parallel discounts and payroll cycles are at their lowest.",
      color: COLORS.accent,
    },
  ];

  return (
    <div>
      <SectionTitle sub="SHAP-inspired feature attributions and prediction explanations">Explainable AI — XAI Module</SectionTitle>

      {/* Feature Importance */}
      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 20, marginBottom: 20 }}>
        <Card>
          <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 20 }}>
            Global Feature Importance — Retail Model
          </div>
          {featureImportance.map((f, i) => (
            <div key={i} style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <span style={{ fontSize: 13, color: COLORS.text.secondary }}>{f.feature}</span>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{
                    fontSize: 11, padding: "2px 8px", borderRadius: 4,
                    background: f.direction === "positive" ? `${COLORS.success}22` : `${COLORS.accent}22`,
                    color: f.direction === "positive" ? COLORS.success : COLORS.accent,
                    fontWeight: 700,
                  }}>{f.direction === "positive" ? "▲ +" : "▼ −"}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, fontFamily: "'Space Mono', monospace" }}>
                    {(f.importance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div style={{ background: COLORS.bg.dark, borderRadius: 4, height: 8 }}>
                <div style={{
                  width: `${f.importance * 400}%`, height: "100%", borderRadius: 4,
                  background: f.direction === "positive"
                    ? `linear-gradient(90deg, ${COLORS.success}, ${COLORS.primary})`
                    : `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.secondary})`,
                  transition: "width 0.8s ease",
                }} />
              </div>
            </div>
          ))}
          <div style={{ marginTop: 14, fontSize: 12, color: COLORS.text.muted, fontStyle: "italic" }}>
            Importance values reflect mean |SHAP| across 10,000 test predictions
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card>
            <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginBottom: 14 }}>Model Confidence Distribution</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={[
                { range: "50–60%", count: 12 }, { range: "60–70%", count: 28 },
                { range: "70–80%", count: 47 }, { range: "80–90%", count: 68 },
                { range: "90–95%", count: 89 }, { range: "95–100%", count: 134 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                <XAxis dataKey="range" tick={{ fill: COLORS.text.muted, fontSize: 10 }} />
                <YAxis tick={{ fill: COLORS.text.muted, fontSize: 10 }} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Bar dataKey="count" fill={COLORS.primary} radius={[3, 3, 0, 0]} name="Predictions" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card>
            <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginBottom: 12 }}>Quick Metrics</div>
            {[
              { label: "Avg Confidence", val: "94.2%" },
              { label: "Low Conf Preds", val: "2.8%" },
              { label: "Explanations/day", val: "~12k" },
              { label: "XAI Method", val: "SHAP TreeExplainer" },
            ].map((m, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${COLORS.bg.border}`, fontSize: 13 }}>
                <span style={{ color: COLORS.text.muted }}>{m.label}</span>
                <span style={{ color: COLORS.primary, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>{m.val}</span>
              </div>
            ))}
          </Card>
        </div>
      </div>

      {/* Explanation Cards */}
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>
        "Why This Prediction?" — Sample Explanation Cards
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
        {explanationCards.map(card => (
          <div key={card.id} onClick={() => setSelected(selected === card.id ? null : card.id)}
            style={{
              background: COLORS.bg.card, border: `1px solid ${selected === card.id ? card.color : COLORS.bg.border}`,
              borderRadius: 16, padding: 20, cursor: "pointer", transition: "all 0.2s",
              boxShadow: selected === card.id ? `0 0 20px ${card.color}33` : "none",
            }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary }}>{card.prediction}</div>
              <div style={{
                fontSize: 13, fontWeight: 900, color: card.color, fontFamily: "'Space Mono', monospace",
                background: `${card.color}18`, borderRadius: 8, padding: "4px 10px",
              }}>{card.confidence}%</div>
            </div>
            <div style={{ marginBottom: 12 }}>
              {card.top_factors.map((f, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: card.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 12, color: COLORS.text.secondary }}>{f}</span>
                </div>
              ))}
            </div>
            {selected === card.id && (
              <div style={{ borderTop: `1px solid ${COLORS.bg.border}`, paddingTop: 12, fontSize: 13, color: COLORS.text.muted, lineHeight: 1.6 }}>
                {card.rationale}
              </div>
            )}
            <div style={{ fontSize: 11, color: card.color, marginTop: 8, fontWeight: 600 }}>
              {selected === card.id ? "▲ Click to collapse" : "▼ Click for full rationale"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Forecast Visualization Page ──────────────────────────────────────────────
const ForecastPage = () => {
  const [view, setView] = useState("daily");

  const seasonalData = [
    { month: "Jan", retail: 0.72, electricity: 1.18 },
    { month: "Feb", retail: 0.68, electricity: 1.12 },
    { month: "Mar", retail: 0.82, electricity: 0.98 },
    { month: "Apr", retail: 0.88, electricity: 0.92 },
    { month: "May", retail: 1.02, electricity: 0.96 },
    { month: "Jun", retail: 1.18, electricity: 1.24 },
    { month: "Jul", retail: 1.08, electricity: 1.28 },
    { month: "Aug", retail: 1.12, electricity: 1.22 },
    { month: "Sep", retail: 0.94, electricity: 1.05 },
    { month: "Oct", retail: 1.06, electricity: 0.99 },
    { month: "Nov", retail: 1.38, electricity: 1.08 },
    { month: "Dec", retail: 1.74, electricity: 1.14 },
  ];

  const weatherData = [
    { temp: 15, sales: 4200 }, { temp: 18, sales: 4600 }, { temp: 22, sales: 5100 },
    { temp: 26, sales: 5800 }, { temp: 30, sales: 6200 }, { temp: 34, sales: 5900 },
    { temp: 38, sales: 5400 }, { temp: 40, sales: 4900 }, { temp: 42, sales: 4500 },
  ];

  return (
    <div>
      <SectionTitle sub="Daily, weekly, monthly views with seasonal and weather analytics">Forecast Visualizations</SectionTitle>

      <div style={{ display: "flex", gap: 6, marginBottom: 24 }}>
        {["daily", "weekly", "monthly", "seasonal"].map(v => (
          <button key={v} onClick={() => setView(v)} style={{
            padding: "6px 18px", borderRadius: 8, border: `1px solid ${view === v ? COLORS.primary : COLORS.bg.border}`,
            background: view === v ? `${COLORS.primary}22` : "transparent",
            color: view === v ? COLORS.primary : COLORS.text.secondary,
            cursor: "pointer", fontSize: 13, fontWeight: 600, textTransform: "capitalize",
          }}>{v}</button>
        ))}
      </div>

      {view === "seasonal" ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Seasonal Index — Retail & Electricity</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={seasonalData}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                <XAxis dataKey="month" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <YAxis domain={[0.5, 2.0]} tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <Tooltip contentStyle={customTooltipStyle} formatter={v => v.toFixed(2)} />
                <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
                <Line dataKey="retail" stroke={COLORS.primary} strokeWidth={2.5} dot={{ r: 4 }} name="Retail Index" />
                <Line dataKey="electricity" stroke={COLORS.warning} strokeWidth={2.5} dot={{ r: 4 }} name="Electricity Index" />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 12, color: COLORS.text.muted, marginTop: 8, fontStyle: "italic" }}>
              Index &gt; 1.0 = above annual average. Nov–Dec retail surges 38–74%.
            </div>
          </Card>

          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Temperature vs Retail Sales</div>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={weatherData}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                <XAxis dataKey="temp" tick={{ fill: COLORS.text.muted, fontSize: 11 }} label={{ value: "Temp °C", position: "insideBottomRight", fill: COLORS.text.muted, fontSize: 11 }} />
                <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <Tooltip contentStyle={customTooltipStyle} formatter={v => `₹${v.toLocaleString()}`} />
                <Area type="monotone" dataKey="sales" stroke={COLORS.secondary} fill={`${COLORS.secondary}33`} strokeWidth={2} name="Avg Sales (₹)" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </div>
      ) : (
        <div style={{ display: "grid", gap: 20 }}>
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>
              {view.charAt(0).toUpperCase() + view.slice(1)} Retail Forecast with Confidence Intervals
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={view === "monthly" ? monthlyRetail : retailDailyData}>
                <defs>
                  <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.15} />
                    <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
                <XAxis dataKey={view === "monthly" ? "month" : "day"} tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
                <Area type="monotone" dataKey="actual" stroke={COLORS.primary} fill="url(#confGrad)" strokeWidth={2} name="Actual" />
                <Line type="monotone" dataKey={view === "monthly" ? "forecast" : "predicted"} stroke={COLORS.accent} strokeWidth={2} strokeDasharray="5 5" name="Forecast" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            {[
              { label: "MAE", val: "₹124", sub: "Mean Absolute Error" },
              { label: "MAPE", val: "2.9%", sub: "Mean Abs % Error" },
              { label: "R²", val: "0.982", sub: "Coefficient of Determination" },
            ].map((m, i) => (
              <div key={i} style={{
                background: COLORS.bg.dark, borderRadius: 12, padding: "16px 18px", textAlign: "center",
                border: `1px solid ${COLORS.bg.border}`,
              }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: COLORS.primary, fontFamily: "'Space Mono', monospace" }}>{m.val}</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text.primary, marginTop: 4 }}>{m.label}</div>
                <div style={{ fontSize: 11, color: COLORS.text.muted, marginTop: 2 }}>{m.sub}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ─── AI Models Comparison Page ────────────────────────────────────────────────
const ModelsPage = () => (
  <div>
    <SectionTitle sub="Traditional vs hybrid AI — accuracy, speed, and explainability benchmarks">AI Model Comparison</SectionTitle>

    {/* Model cards */}
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16, marginBottom: 28 }}>
      {modelComparison.map((m, i) => {
        const isHybrid = m.name === "Hybrid (Ours)";
        return (
          <div key={i} style={{
            background: COLORS.bg.card, border: `${isHybrid ? 2 : 1}px solid ${isHybrid ? COLORS.primary : COLORS.bg.border}`,
            borderRadius: 16, padding: 20, position: "relative", overflow: "hidden",
          }}>
            {isHybrid && (
              <div style={{
                position: "absolute", top: 10, right: 10, background: `${COLORS.primary}22`,
                color: COLORS.primary, fontSize: 10, padding: "2px 8px", borderRadius: 4, fontWeight: 700,
              }}>BEST</div>
            )}
            <div style={{ fontSize: 16, fontWeight: 900, color: COLORS.text.primary, marginBottom: 16 }}>{m.name}</div>
            {[
              { label: "Accuracy", val: `${m.accuracy}%`, color: COLORS.success },
              { label: "MAPE", val: `${m.mape}%`, color: m.mape < 4 ? COLORS.success : m.mape < 6 ? COLORS.warning : COLORS.accent },
              { label: "RMSE", val: m.rmse, color: COLORS.primary },
              { label: "Latency (ms)", val: m.latency, color: COLORS.secondary },
            ].map((stat, j) => (
              <div key={j} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: `1px solid ${COLORS.bg.border}` }}>
                <span style={{ fontSize: 12, color: COLORS.text.muted }}>{stat.label}</span>
                <span style={{ fontSize: 13, fontWeight: 800, color: stat.color, fontFamily: "'Space Mono', monospace" }}>{stat.val}</span>
              </div>
            ))}
            <div style={{ marginTop: 12, background: COLORS.bg.dark, borderRadius: 6, padding: "6px 10px" }}>
              <div style={{ fontSize: 11, color: COLORS.text.muted, marginBottom: 3 }}>Accuracy</div>
              <div style={{ background: `${COLORS.bg.border}`, borderRadius: 3, height: 6 }}>
                <div style={{ width: `${m.accuracy}%`, height: "100%", borderRadius: 3, background: isHybrid ? `linear-gradient(90deg, ${COLORS.primary}, ${COLORS.secondary})` : COLORS.text.muted }} />
              </div>
            </div>
          </div>
        );
      })}
    </div>

    {/* Radar chart */}
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
      <Card>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Multi-Dimensional Performance Radar</div>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData}>
            <PolarGrid stroke={COLORS.bg.border} />
            <PolarAngleAxis dataKey="subject" tick={{ fill: COLORS.text.secondary, fontSize: 12 }} />
            <Radar name="ARIMA" dataKey="ARIMA" stroke={COLORS.text.muted} fill={COLORS.text.muted} fillOpacity={0.1} />
            <Radar name="LSTM" dataKey="LSTM" stroke={COLORS.secondary} fill={COLORS.secondary} fillOpacity={0.15} />
            <Radar name="Hybrid" dataKey="Hybrid" stroke={COLORS.primary} fill={COLORS.primary} fillOpacity={0.25} />
            <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
            <Tooltip contentStyle={customTooltipStyle} />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>MAPE Comparison by Model</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={modelComparison}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bg.border} />
            <XAxis dataKey="name" tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
            <YAxis tick={{ fill: COLORS.text.muted, fontSize: 11 }} />
            <Tooltip contentStyle={customTooltipStyle} formatter={v => `${v}%`} />
            <Legend wrapperStyle={{ fontSize: 12, color: COLORS.text.secondary }} />
            <Bar dataKey="mape" name="MAPE %" radius={[4, 4, 0, 0]}>
              {modelComparison.map((m, i) => (
                <Cell key={i} fill={m.name === "Hybrid (Ours)" ? COLORS.primary : CHART_COLORS[i + 1]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  </div>
);

// ─── Settings Page ────────────────────────────────────────────────────────────
const SettingsPage = () => {
  const [settings, setSettings] = useState({
    notifications: true, darkMode: true, autoRetrain: false,
    forecastHorizon: 30, confidenceLevel: 95, updateFreq: "hourly",
  });

  const toggle = key => setSettings(s => ({ ...s, [key]: !s[key] }));

  const Toggle = ({ k }) => (
    <div onClick={() => toggle(k)} style={{
      width: 44, height: 24, borderRadius: 12, background: settings[k] ? COLORS.primary : COLORS.bg.border,
      position: "relative", cursor: "pointer", transition: "background 0.2s", flexShrink: 0,
    }}>
      <div style={{
        width: 18, height: 18, borderRadius: "50%", background: "#fff",
        position: "absolute", top: 3, left: settings[k] ? 23 : 3, transition: "left 0.2s",
      }} />
    </div>
  );

  return (
    <div style={{ maxWidth: 680 }}>
      <SectionTitle sub="Platform configuration and user preferences">Settings</SectionTitle>

      {/* Profile */}
      <Card style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>User Profile</div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 64, height: 64, borderRadius: "50%",
            background: `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.secondary})`,
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, fontWeight: 900, color: "#fff",
          }}>A</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: COLORS.text.primary }}>Admin User</div>
            <div style={{ fontSize: 13, color: COLORS.text.secondary }}>admin@smartxsense.ai</div>
            <GlowBadge color={COLORS.success}>Analyst Pro</GlowBadge>
          </div>
          <button style={{
            marginLeft: "auto", padding: "8px 20px", borderRadius: 8, border: `1px solid ${COLORS.bg.border}`,
            background: "transparent", color: COLORS.text.secondary, cursor: "pointer", fontSize: 13,
          }}>Edit Profile</button>
        </div>
      </Card>

      {/* Preferences */}
      <Card style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Preferences</div>
        {[
          { label: "Enable Notifications", sub: "Receive alerts for anomalies and high-confidence predictions", key: "notifications" },
          { label: "Dark Mode", sub: "Use the dark UI theme (active)", key: "darkMode" },
          { label: "Auto-Retrain Models", sub: "Automatically retrain when new 30-day data is available", key: "autoRetrain" },
        ].map((s, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "14px 0", borderBottom: `1px solid ${COLORS.bg.border}` }}>
            <div>
              <div style={{ fontSize: 14, color: COLORS.text.primary, fontWeight: 600 }}>{s.label}</div>
              <div style={{ fontSize: 12, color: COLORS.text.muted, marginTop: 2 }}>{s.sub}</div>
            </div>
            <Toggle k={s.key} />
          </div>
        ))}
      </Card>

      {/* Model Config */}
      <Card>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.text.primary, marginBottom: 16 }}>Model Configuration</div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, color: COLORS.text.secondary, marginBottom: 6 }}>Forecast Horizon: {settings.forecastHorizon} days</div>
          <input type="range" min={7} max={90} step={1} value={settings.forecastHorizon}
            onChange={e => setSettings(s => ({ ...s, forecastHorizon: Number(e.target.value) }))}
            style={{ width: "100%", accentColor: COLORS.primary }} />
        </div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, color: COLORS.text.secondary, marginBottom: 6 }}>Confidence Level: {settings.confidenceLevel}%</div>
          <input type="range" min={80} max={99} step={1} value={settings.confidenceLevel}
            onChange={e => setSettings(s => ({ ...s, confidenceLevel: Number(e.target.value) }))}
            style={{ width: "100%", accentColor: COLORS.primary }} />
        </div>
        <div>
          <div style={{ fontSize: 13, color: COLORS.text.secondary, marginBottom: 6 }}>Update Frequency</div>
          <select value={settings.updateFreq} onChange={e => setSettings(s => ({ ...s, updateFreq: e.target.value }))}
            style={{
              background: COLORS.bg.dark, border: `1px solid ${COLORS.bg.border}`, borderRadius: 8,
              padding: "8px 12px", color: COLORS.text.primary, fontSize: 13, outline: "none", width: "100%",
            }}>
            <option value="realtime">Real-time (streaming)</option>
            <option value="hourly">Hourly</option>
            <option value="daily">Daily batch</option>
          </select>
        </div>
      </Card>
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
    dashboard: <DashboardPage />,
    retail: <RetailPage />,
    electricity: <ElectricityPage />,
    xai: <XAIPage />,
    forecast: <ForecastPage />,
    models: <ModelsPage />,
    settings: <SettingsPage />,
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
