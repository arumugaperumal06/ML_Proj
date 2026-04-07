import React, { useState, useCallback } from "react";
import axios from "axios";
import "./Dashboard.css";

const API_URL = "http://127.0.0.1:5000/predict";

const FIELDS = [
  {
    name: "temperature",
    label: "Temperature",
    unit: "K",
    placeholder: "e.g. 300",
    icon: "🌡️",
    min: 200,
    max: 400,
  },
  {
    name: "vibration",
    label: "Vibration",
    unit: "rpm",
    placeholder: "e.g. 1200",
    icon: "📳",
    min: 0,
    max: 3000,
  },
  {
    name: "pressure",
    label: "Torque / Pressure",
    unit: "Nm",
    placeholder: "e.g. 30",
    icon: "⚙️",
    min: 0,
    max: 100,
  },
  {
    name: "runtime",
    label: "Tool Wear",
    unit: "min",
    placeholder: "e.g. 200",
    icon: "⏱️",
    min: 0,
    max: 2000,
  },
];

const STATUS_META = {
  SAFE:    { color: "#22c55e", bg: "rgba(34,197,94,0.12)",    icon: "✅", label: "System Healthy" },
  WARNING: { color: "#f59e0b", bg: "rgba(245,158,11,0.12)",   icon: "⚠️", label: "Attention Required" },
  FAILURE: { color: "#ef4444", bg: "rgba(239,68,68,0.12)",    icon: "🚨", label: "Failure Risk Detected" },
};

function GaugeBar({ probability }) {
  const pct = Math.round(probability * 100);
  const color =
    pct < 30 ? "#22c55e" :
    pct < 65 ? "#f59e0b" :
               "#ef4444";

  return (
    <div className="gauge-wrap">
      <div className="gauge-label">
        <span>Failure Probability</span>
        <span style={{ color }} className="gauge-pct">{pct}%</span>
      </div>
      <div className="gauge-track">
        <div
          className="gauge-fill"
          style={{ width: `${pct}%`, background: color }}
        />
        <div className="gauge-marker" style={{ left: "30%" }} title="Warning threshold" />
        <div className="gauge-marker" style={{ left: "65%" }} title="Failure threshold" />
      </div>
      <div className="gauge-legend">
        <span>Safe</span>
        <span>Warning</span>
        <span>Failure</span>
      </div>
    </div>
  );
}

function HealthRing({ score }) {
  const r = 42;
  const circ = 2 * Math.PI * r;
  const fill = circ * (score / 100);
  const color =
    score > 65 ? "#22c55e" :
    score > 35 ? "#f59e0b" :
                 "#ef4444";

  return (
    <div className="health-ring">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" />
        <circle
          cx="55" cy="55" r={r}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={`${fill} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 55 55)"
          style={{ transition: "stroke-dasharray 0.8s ease" }}
        />
      </svg>
      <div className="health-ring-center">
        <span className="health-score" style={{ color }}>{score.toFixed(0)}</span>
        <span className="health-label">Health</span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [formData, setFormData] = useState({
    temperature: "",
    vibration: "",
    pressure: "",
    runtime: "",
  });
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const handleChange = useCallback((e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear stale results when inputs change so user knows to re-predict
    setResult(null);
    setError(null);
  }, []);

  const isFormValid = FIELDS.every((f) => formData[f.name] !== "" && !isNaN(Number(formData[f.name])));

  const handleSubmit = async () => {
    if (!isFormValid) {
      setError("Please fill in all fields with valid numeric values.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = Object.fromEntries(
        FIELDS.map((f) => [f.name, Number(formData[f.name])])
      );

      const res = await axios.post(API_URL, payload, {
        headers: { "Content-Type": "application/json" },
        timeout: 10000,
      });

      setResult(res.data);
    } catch (err) {
      if (err.code === "ECONNABORTED") {
        setError("Request timed out. Is the backend server running?");
      } else if (err.response) {
        setError(err.response.data?.error || `Server error: ${err.response.status}`);
      } else {
        setError("Cannot reach the backend. Make sure Flask is running on port 5000.");
      }
    } finally {
      setLoading(false);
    }
  };

  const statusMeta = result ? STATUS_META[result.status] : null;

  return (
    <div className="pm-root">
      {/* Header */}
      <header className="pm-header">
        <div className="pm-header-inner">
          <div className="pm-logo">
            <span className="pm-logo-icon">⚙️</span>
            <div>
              <div className="pm-logo-title">PredictiveMaint</div>
              <div className="pm-logo-sub">ML-Powered Equipment Monitor</div>
            </div>
          </div>
          <div className="pm-header-badge">
            <span className="pm-status-dot" />
            API Connected
          </div>
        </div>
      </header>

      <main className="pm-main">
        {/* Input Card */}
        <section className="pm-card pm-input-card">
          <div className="pm-card-header">
            <h2 className="pm-card-title">Sensor Readings</h2>
            <p className="pm-card-sub">Enter current machine parameters to predict maintenance needs</p>
          </div>

          <div className="pm-fields">
            {FIELDS.map((field) => (
              <div className="pm-field" key={field.name}>
                <label className="pm-field-label" htmlFor={field.name}>
                  <span className="pm-field-icon">{field.icon}</span>
                  {field.label}
                  <span className="pm-field-unit">{field.unit}</span>
                </label>
                <input
                  id={field.name}
                  name={field.name}
                  type="number"
                  className="pm-input"
                  placeholder={field.placeholder}
                  value={formData[field.name]}
                  onChange={handleChange}
                  min={field.min}
                  max={field.max}
                  autoComplete="off"
                />
              </div>
            ))}
          </div>

          <button
            className={`pm-btn ${loading ? "pm-btn-loading" : ""}`}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="pm-spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <span>🔬</span> Run Prediction
              </>
            )}
          </button>

          {/* Error state */}
          {error && (
            <div className="pm-error">
              <span>❌</span> {error}
            </div>
          )}
        </section>

        {/* Results Card — only shown when we have results */}
        {result && statusMeta && (
          <section className="pm-card pm-result-card" style={{ borderColor: statusMeta.color }}>
            {/* Status Banner */}
            <div className="pm-status-banner" style={{ background: statusMeta.bg }}>
              <span className="pm-status-icon">{statusMeta.icon}</span>
              <div>
                <div className="pm-status-title" style={{ color: statusMeta.color }}>
                  {result.status}
                </div>
                <div className="pm-status-desc">{statusMeta.label}</div>
              </div>
            </div>

            {/* Metrics Row */}
            <div className="pm-metrics">
              <HealthRing score={result.health_score ?? (1 - result.probability) * 100} />

              <div className="pm-metrics-right">
                <GaugeBar probability={result.probability} />
              </div>
            </div>

            {/* Contributions */}
            {result.contributions && (
              <div className="pm-contribs">
                <h3 className="pm-section-title">Feature Contributions</h3>
                <div className="pm-contrib-bars">
                  {Object.entries(result.contributions)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .map(([feat, val]) => {
                      const isPos = val >= 0;
                      const pct   = Math.min(Math.abs(val) * 400, 100);
                      return (
                        <div className="pm-contrib-row" key={feat}>
                          <span className="pm-contrib-name">{feat}</span>
                          <div className="pm-contrib-track">
                            <div
                              className="pm-contrib-fill"
                              style={{
                                width: `${pct}%`,
                                background: isPos ? "#ef4444" : "#22c55e",
                              }}
                            />
                          </div>
                          <span
                            className="pm-contrib-val"
                            style={{ color: isPos ? "#ef4444" : "#22c55e" }}
                          >
                            {isPos ? "+" : ""}{val.toFixed(3)}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

            {/* Reasons */}
            <div className="pm-reasons">
              <h3 className="pm-section-title">🧠 Key Factors</h3>
              <ul className="pm-reason-list">
                {result.reasons.map((r, i) => (
                  <li key={i} className="pm-reason-item">
                    <span className="pm-reason-num">{i + 1}</span>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          </section>
        )}

        {/* Placeholder when no result yet */}
        {!result && !loading && !error && (
          <section className="pm-card pm-empty-card">
            <div className="pm-empty">
              <span className="pm-empty-icon">📊</span>
              <p>Enter sensor readings and click <strong>Run Prediction</strong> to see the analysis.</p>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}