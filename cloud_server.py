"""
CLOUD SERVER WITH FULL DASHBOARD - Runs on Railway
Two-layer severity system with alert panel
"""

from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

all_readings  = []
zone_data     = {}
global_model  = None
global_scaler = StandardScaler()


@app.route('/send-data', methods=['POST'])
def receive_data():
    global global_model
    data    = request.get_json()
    zone    = data.get("zone", "Unknown")
    reading = {
        "zone":          zone,
        "timestamp":     data.get("timestamp"),
        "ph":            data.get("ph"),
        "tds":           data.get("tds"),
        "turbidity":     data.get("turbidity") or data.get("ntu"),
        "wqi":           data.get("wqi"),
        "wqi_flag":      data.get("wqi_flag", 0),
        "prediction":    data.get("prediction", 1),
        "anomaly_score": data.get("anomaly_score", 0),
        "streak":        data.get("streak", 0),
        "severity":      data.get("severity", "NORMAL"),
        "anomaly":       data.get("anomaly", "NORMAL"),
        "received_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    all_readings.append(reading)
    if zone not in zone_data:
        zone_data[zone] = []
    zone_data[zone].append([
        data.get("ph", 0), data.get("tds", 0),
        data.get("turbidity") or data.get("ntu") or 0,
        data.get("wqi", 0)
    ])
    print(f"[{reading['received_at']}] {zone} | Severity={reading['severity']} Score={reading['anomaly_score']}")
    if len(zone_data) >= 2:
        build_global_model()
    return jsonify({"status": "success", "message": f"Data from {zone} received"}), 200


@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({
        "total_readings":     len(all_readings),
        "zones_connected":    list(zone_data.keys()),
        "global_model_ready": global_model is not None,
        "readings":           all_readings[-20:]
    }), 200


@app.route('/global-model', methods=['GET'])
def get_global_model():
    if global_model is None:
        return jsonify({"status": "not ready", "message": f"Need at least 2 zones. Currently have: {list(zone_data.keys())}"}), 200
    return jsonify({"status": "ready", "zones_used": list(zone_data.keys()), "total_readings": len(all_readings), "message": "Global model trained and ready!"}), 200


@app.route('/clear', methods=['GET'])
def clear_data():
    global all_readings, zone_data, global_model
    all_readings = []
    zone_data    = {}
    global_model = None
    return jsonify({"status": "success", "message": "All data cleared. Total readings: 0"}), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "project": "Critical Water Infrastructure Anomaly Detection",
        "status":  "Cloud server running",
        "endpoints": {
            "POST /send-data":   "Send sensor data from laptop",
            "GET /data":         "View all stored readings (JSON)",
            "GET /dashboard":    "Live visual dashboard",
            "GET /global-model": "Check global model status",
            "GET /clear":        "Clear all stored data"
        }
    }), 200


@app.route('/dashboard', methods=['GET'])
def dashboard():
    readings  = all_readings
    n_total   = len(readings)
    zones     = list(zone_data.keys())
    gm_status = "Ready" if global_model is not None else "Waiting for 2+ zones"
    gm_color  = "#02C39A" if global_model is not None else "#FF8C00"
    gm_bg     = "#003D2B" if global_model is not None else "#3D1A00"

    # Severity counts
    from collections import Counter
    sev_counts = Counter(r.get("severity", "NORMAL") for r in readings)

    # Alert panels — last 10 of each type
    wqi_alerts      = [r for r in readings if r.get("wqi_flag") == 1][-10:]
    behav_alerts    = [r for r in readings if r.get("prediction") == -1][-10:]
    critical_alerts = [r for r in readings if r.get("severity") == "CRITICAL"][-10:]
    caution_alerts  = [r for r in readings if r.get("severity") == "CAUTION"][-10:]

    # Chart data
    labels    = [r.get("timestamp", "")[-8:] for r in readings]
    ph_vals   = [r.get("ph")        or 0 for r in readings]
    tds_vals  = [r.get("tds")       or 0 for r in readings]
    turb_vals = [r.get("turbidity") or 0 for r in readings]
    wqi_vals  = [r.get("wqi")       or 0 for r in readings]
    severities = [r.get("severity", "NORMAL") for r in readings]

    sev_color_map = {
        "NORMAL":   "#02C39A",
        "WATCH":    "#4FC3F7",
        "CAUTION":  "#FF8C00",
        "WARNING":  "#FFD700",
        "CRITICAL": "#FF4444"
    }

    # Row severity colors
    row_bg_map = {
        "NORMAL":   "",
        "WATCH":    'style="background:#0D1F3C;color:#4FC3F7;"',
        "CAUTION":  'style="background:#3D1A00;color:#FF8C00;"',
        "WARNING":  'style="background:#3D3000;color:#FFD700;"',
        "CRITICAL": 'style="background:#3D0000;color:#FF6B6B;"'
    }
    badge_map = {
        "NORMAL":   '<span style="background:#006644;padding:2px 8px;border-radius:8px;font-size:10px;color:#fff;">NORMAL</span>',
        "WATCH":    '<span style="background:#0D47A1;padding:2px 8px;border-radius:8px;font-size:10px;color:#fff;">WATCH</span>',
        "CAUTION":  '<span style="background:#BF360C;padding:2px 8px;border-radius:8px;font-size:10px;color:#fff;">CAUTION</span>',
        "WARNING":  '<span style="background:#F57F17;padding:2px 8px;border-radius:8px;font-size:10px;color:#000;">WARNING</span>',
        "CRITICAL": '<span style="background:#B71C1C;padding:2px 8px;border-radius:8px;font-size:10px;color:#fff;">CRITICAL</span>'
    }

    # Table rows
    table_rows = ""
    for r in reversed(readings[-50:]):
        sev       = r.get("severity", "NORMAL")
        row_style = row_bg_map.get(sev, "")
        badge     = badge_map.get(sev, "")
        table_rows += f"""
        <tr {row_style}>
            <td>{r.get('received_at','')}</td>
            <td>{r.get('zone','')}</td>
            <td>{r.get('ph','')}</td>
            <td>{r.get('tds','')}</td>
            <td>{r.get('turbidity','')}</td>
            <td>{r.get('wqi','')}</td>
            <td>{r.get('wqi_flag','')}</td>
            <td>{r.get('anomaly_score','')}</td>
            <td>{r.get('streak','')}</td>
            <td>{badge}</td>
        </tr>"""

    # Alert panel rows
    def wqi_rows(alerts):
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('wqi','')}</td></tr>"
        return rows or "<tr><td colspan='3' style='color:#64748B;text-align:center;'>No alerts</td></tr>"

    def behav_rows(alerts):
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('anomaly_score','')}</td><td>{r.get('streak','')}</td></tr>"
        return rows or "<tr><td colspan='4' style='color:#64748B;text-align:center;'>No alerts</td></tr>"

    def critical_rows(alerts):
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('ph','')}</td><td>{r.get('tds','')}</td><td>{r.get('wqi','')}</td><td>{r.get('anomaly_score','')}</td></tr>"
        return rows or "<tr><td colspan='6' style='color:#64748B;text-align:center;'>No critical alerts</td></tr>"

    def caution_rows(alerts):
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('anomaly_score','')}</td><td>{r.get('streak','')}</td></tr>"
        return rows or "<tr><td colspan='4' style='color:#64748B;text-align:center;'>No caution alerts — system nominal</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Water Quality Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  *{{ margin:0;padding:0;box-sizing:border-box; }}
  body{{ background:#0A0F1E;color:#E0E6F0;font-family:Arial,sans-serif;padding:20px; }}
  h1{{ text-align:center;color:#02C39A;font-size:22px;margin-bottom:4px;letter-spacing:1px; }}
  .subtitle{{ text-align:center;color:#64748B;font-size:11px;margin-bottom:20px; }}
  .cards{{ display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap; }}
  .card{{ flex:1;min-width:120px;background:#161B2E;border-radius:10px;padding:14px;text-align:center;border:1px solid #1E293B; }}
  .card .num{{ font-size:32px;font-weight:bold; }}
  .card .lbl{{ font-size:11px;color:#64748B;margin-top:4px; }}
  .sev-cards{{ display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap; }}
  .sev-card{{ flex:1;min-width:100px;border-radius:8px;padding:10px;text-align:center;border:1px solid; }}
  .gm-row{{ background:#161B2E;border-radius:10px;padding:14px;margin-bottom:16px;border:1px solid #1E293B;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px; }}
  .gm-label{{ font-size:13px;font-weight:bold;color:#4FC3F7;margin-bottom:3px; }}
  .gm-zones{{ font-size:11px;color:#64748B; }}
  .gm-badge{{ padding:5px 16px;border-radius:20px;font-size:12px;font-weight:bold;background:{gm_bg};color:{gm_color};border:1px solid {gm_color}; }}
  .section-title{{ color:#4FC3F7;font-size:14px;font-weight:bold;margin:16px 0 8px 0; }}
  .caution-banner{{ background:#1A0D00;border:1px solid #FF8C00;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:12px;color:#FF8C00;display:{"block" if caution_alerts else "none"}; }}
  .caution-banner strong{{ font-size:14px; }}
  .alert-panels{{ display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px; }}
  .alert-box{{ background:#161B2E;border-radius:10px;padding:14px;border:1px solid; }}
  .alert-box h3{{ font-size:12px;font-weight:bold;margin-bottom:8px; }}
  .alert-box table{{ width:100%;border-collapse:collapse; }}
  .alert-box td{{ padding:5px 6px;font-size:10px;border-bottom:1px solid #1E293B; }}
  .alert-box th{{ padding:5px 6px;font-size:10px;font-weight:bold; }}
  .charts{{ display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px; }}
  .chart-box{{ background:#161B2E;border-radius:10px;padding:14px;border:1px solid #1E293B; }}
  .chart-title{{ color:#4FC3F7;font-size:12px;font-weight:bold;margin-bottom:8px; }}
  table.main{{ width:100%;border-collapse:collapse;background:#161B2E;border-radius:10px;overflow:hidden; }}
  table.main th{{ background:#1C7293;padding:8px 10px;font-size:11px;text-align:left;color:#fff; }}
  table.main td{{ padding:6px 10px;font-size:11px;border-bottom:1px solid #1E293B; }}
  @media(max-width:700px){{ .alert-panels{{grid-template-columns:1fr;}}.charts{{grid-template-columns:1fr;}} }}
</style>
</head>
<body>

<h1>Water Quality Monitoring Dashboard</h1>
<p class="subtitle">Critical Water Infrastructure Anomaly Detection | Batch 20 | Auto-refreshes every 30 seconds</p>

<!-- STAT CARDS -->
<div class="cards">
  <div class="card" style="border-color:#4FC3F7;"><div class="num" style="color:#4FC3F7;">{n_total}</div><div class="lbl">Total Readings</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#FFD700;">{len(zones)}</div><div class="lbl">Zones Connected</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#02C39A;">{sev_counts.get("NORMAL",0)}</div><div class="lbl">Normal</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#4FC3F7;">{sev_counts.get("WATCH",0)}</div><div class="lbl">Watch</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#FF8C00;">{sev_counts.get("CAUTION",0)}</div><div class="lbl">Caution</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#FFD700;">{sev_counts.get("WARNING",0)}</div><div class="lbl">Warning</div></div>
  <div class="card" style="border-color:#FF4444;"><div class="num" style="color:#FF4444;">{sev_counts.get("CRITICAL",0)}</div><div class="lbl">Critical</div></div>
</div>

<!-- GLOBAL MODEL -->
<div class="gm-row">
  <div>
    <div class="gm-label">Global Model - Federated Learning</div>
    <div class="gm-zones">Zones: {', '.join(zones) if zones else 'None connected yet'}</div>
  </div>
  <div class="gm-badge">{gm_status}</div>
</div>

<!-- CAUTION BANNER -->
<div class="caution-banner">
  <strong>CAUTION — Early Warning Active</strong><br>
  Sustained behavioural deviation detected while WQI remains within regulatory limits.
  The ML layer has identified a pattern shift before the threshold was crossed.
  This is the value of behavioural monitoring over threshold-only systems.
</div>

<!-- THREE ALERT PANELS -->
<div class="section-title">Alert Panels</div>
<div class="alert-panels">

  <div class="alert-box" style="border-color:#FF8C00;">
    <h3 style="color:#FF8C00;">REGULATORY BREACH (WQI Flag = 1)</h3>
    <table>
      <tr><th style="color:#FF8C00;">Time</th><th style="color:#FF8C00;">Zone</th><th style="color:#FF8C00;">WQI</th></tr>
      {wqi_rows(wqi_alerts)}
    </table>
  </div>

  <div class="alert-box" style="border-color:#4FC3F7;">
    <h3 style="color:#4FC3F7;">BEHAVIOURAL ANOMALY (Prediction = -1)</h3>
    <table>
      <tr><th style="color:#4FC3F7;">Time</th><th style="color:#4FC3F7;">Zone</th><th style="color:#4FC3F7;">Score</th><th style="color:#4FC3F7;">Streak</th></tr>
      {behav_rows(behav_alerts)}
    </table>
  </div>

  <div class="alert-box" style="border-color:#FF4444;">
    <h3 style="color:#FF4444;">CRITICAL — CONFIRMED EVENT</h3>
    <table>
      <tr><th style="color:#FF4444;">Time</th><th style="color:#FF4444;">Zone</th><th style="color:#FF4444;">pH</th><th style="color:#FF4444;">TDS</th><th style="color:#FF4444;">WQI</th><th style="color:#FF4444;">Score</th></tr>
      {critical_rows(critical_alerts)}
    </table>
  </div>

</div>

<!-- CAUTION PANEL -->
<div class="section-title">Early Warning — Caution Events (ML catches what threshold misses)</div>
<div class="alert-box" style="border-color:#FF8C00;margin-bottom:16px;">
  <h3 style="color:#FF8C00;">CAUTION — Sustained Behavioural Deviation, WQI Still Compliant</h3>
  <table style="width:100%">
    <tr><th style="color:#FF8C00;">Time</th><th style="color:#FF8C00;">Zone</th><th style="color:#FF8C00;">Score</th><th style="color:#FF8C00;">Streak</th></tr>
    {caution_rows(caution_alerts)}
  </table>
</div>

<!-- CHARTS -->
<div class="section-title">Sensor Trends</div>
<div class="charts">
  <div class="chart-box"><div class="chart-title">pH Over Time</div><canvas id="phChart" height="130"></canvas></div>
  <div class="chart-box"><div class="chart-title">WQI Over Time</div><canvas id="wqiChart" height="130"></canvas></div>
  <div class="chart-box"><div class="chart-title">TDS Over Time</div><canvas id="tdsChart" height="130"></canvas></div>
  <div class="chart-box"><div class="chart-title">Turbidity (NTU) Over Time</div><canvas id="turbChart" height="130"></canvas></div>
</div>

<!-- FULL TABLE -->
<div class="section-title">All Readings (Last 50)</div>
<table class="main">
  <thead>
    <tr>
      <th>Timestamp</th><th>Zone</th><th>pH</th><th>TDS</th><th>NTU</th>
      <th>WQI</th><th>WQI Flag</th><th>Score</th><th>Streak</th><th>Severity</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<script>
const labels    = {labels};
const phVals    = {ph_vals};
const tdsVals   = {tds_vals};
const turbVals  = {turb_vals};
const wqiVals   = {wqi_vals};
const sevs      = {severities};

const colorMap  = {{
  NORMAL:'#02C39A', WATCH:'#4FC3F7',
  CAUTION:'#FF8C00', WARNING:'#FFD700', CRITICAL:'#FF4444'
}};
const ptColors  = sevs.map(s => colorMap[s] || '#02C39A');

function makeChart(id, label, data, color) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [{{
        label: label, data: data,
        borderColor: color, backgroundColor: color + '22',
        pointBackgroundColor: ptColors,
        pointRadius: 4, tension: 0.3, fill: true
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ labels: {{ color:'#E0E6F0', font:{{ size:10 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color:'#64748B', maxTicksLimit:8, font:{{ size:9 }} }}, grid:{{ color:'#1E293B' }} }},
        y: {{ ticks: {{ color:'#64748B', font:{{ size:9  }} }}, grid:{{ color:'#1E293B' }} }}
      }}
    }}
  }});
}}

makeChart('phChart',   'pH',        phVals,   '#4FC3F7');
makeChart('wqiChart',  'WQI',       wqiVals,  '#02C39A');
makeChart('tdsChart',  'TDS (ppm)', tdsVals,  '#FFD700');
makeChart('turbChart', 'Turbidity', turbVals, '#FF8C00');
</script>
</body>
</html>"""
    return html


def build_global_model():
    global global_model, global_scaler
    all_zone_data = []
    for zone, readings in zone_data.items():
        all_zone_data.extend(readings)
    combined      = np.array(all_zone_data)
    global_scaler = StandardScaler()
    scaled        = global_scaler.fit_transform(combined)
    global_model  = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    global_model.fit(scaled)
    print(f"[CLOUD] Global Model ready! Zones: {list(zone_data.keys())} | Samples: {len(combined)}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
