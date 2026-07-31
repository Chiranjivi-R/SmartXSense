# SmartXSense
### Explainable Hybrid Demand Forecasting for Retail & Electricity Systems

A futuristic AI-powered frontend dashboard built with React.js and Recharts.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
npm install
```

### 2. Run development server
```bash
npm run dev
```

### 3. Open in browser
```
http://localhost:5173
```

---

## 📦 Tech Stack

| Technology | Purpose |
|---|---|
| React.js 18 | UI framework |
| Recharts | Charts and graphs |
| Vite | Build tool & dev server |
| Google Fonts | Space Grotesk + Space Mono |

---

## 📁 Project Structure

```
smartxsense/
├── index.html          → App entry HTML
├── package.json        → Dependencies
├── vite.config.js      → Vite configuration
├── .gitignore          → Git ignore rules
├── README.md           → This file
└── src/
    ├── main.jsx        → React entry point
    ├── index.css       → Global styles
    └── App.jsx         → Main application (all pages & components)
```

---

## 🎨 Pages

| Page | Description |
|---|---|
| 🏠 Home | Landing page with hero, features, testimonials |
| 📊 Dashboard | Analytics overview with KPI cards and charts |
| 🛒 Retail | Product demand, inventory, sales analytics |
| ⚡ Electricity | Power consumption, grid status, peak analysis |
| 🔍 Explainability | SHAP feature importance, XAI insights |
| 📈 Forecasts | Daily/weekly/monthly/seasonal forecast views |
| 🤖 AI Models | Model comparison — ARIMA vs LSTM vs SmartXSense |
| ⚙️ Settings | User profile, preferences, model config |

---

## 🎯 Features

- Glassmorphism dark UI with neon glow effects
- Animated loading screen
- Collapsible sidebar navigation
- Live clock in top bar
- Notification panel
- Dark/Light mode toggle
- Responsive charts (Recharts)
- Interactive XAI explanation cards
- Smart grid status panel
- Confidence interval visualization
- Feature importance SHAP bars
- Model comparison radar chart

---

## 📊 Charts Used

- AreaChart — Retail demand forecast
- LineChart — Electricity hourly consumption
- BarChart — Region-wise demand, model comparison
- PieChart — Category share (donut)
- RadarChart — Model performance radar
- ComposedChart — Forecast with confidence intervals

---

## 🏗️ Build for Production

```bash
npm run build
```

Output will be in the `dist/` folder.

---

## 📝 Notes

- All data is static dummy JSON (no backend required)
- Single file architecture — entire app in `src/App.jsx`
- No authentication or database needed
- Fully frontend-only application

---

*SmartXSense — Frontend Demo v1.0*
