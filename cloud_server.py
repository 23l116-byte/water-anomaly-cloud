"""
CLOUD SERVER WITH DASHBOARD - Runs on Railway
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


# -- ROUTE 1: Receive data --
@app.route('/send-data', methods=['POST'])
def receive_data():
    global global_model
    data    = request.get_json()
    zone    = data.get("zone", "Unknown")
    reading = {
        "zone":        zone,
        "timestamp":   data.get("timestamp"),
        "ph":          data.get("ph"),
        "tds":         data.get("tds"),
        "turbidity":   data.get("turbidity") or data.get("ntu"),
        "wqi":         data.get("wqi"),
        "anomaly":     data.get("anomaly"),
        "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    all_readings.append(reading)
    if zone not in zone_data:
        zone_data[zone] = []
    zone_data[zone].append([
        data.get("ph", 0), data.get("tds", 0),
        data.get("turbidity") or data.get("ntu") or 0,
        data.get("wqi", 0)
    ])
    print(f"[{reading['received_at']}] {zone} | pH={reading['ph']} TDS={reading['tds']} WQI={reading['wqi']} Anomaly={reading['anomaly']}")
    if len(zone_data) >= 2:
        build_global_model()
    return jsonify({"status": "success", "message": f"Data from {zone} received"}), 200


# -- ROUTE 2: JSON data --
@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({
        "total_readings":     len(all_readings),
        "zones_connected":    list(zone_data.keys()),
        "global_model_ready": global_model is not None,
        "readings":           all_readings[-20:]
    }), 200


# -- ROUTE 3: Global model --
@app.route('/global-model', methods=['GET'])
def get_global_model():
    if global_model is None:
        return jsonify({"status": "not ready", "message": f"Need at least 2 zones. Currently have: {list(zone_data.keys())}"}), 200
    return jsonify({"status": "ready", "zones_used": list(zone_data.keys()), "total_readings": len(all_readings), "message": "Global model trained and ready!"}), 200


# -- ROUTE 4: Clear data --
@app.route('/clear', methods=['GET'])
def clear_data():
    global all_readings, zone_data, global_model
    all_readings = []
    zone_data    = {}
    global_model = None
    return jsonify({"status": "success", "message": "All data cleared. Total readings: 0"}), 200


# -- ROUTE 5: Home --
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


# -- ROUTE 6: DASHBOARD --
@app.route('/dashboard', methods=['GET'])
def dashboard():
    readings  = all_readings
    n_total   = len(readings)
    n_anomaly = sum(1 for r in readings if r.get("anomaly") == "ANOMALY")
    n_normal  = n_total - n_anomaly
    zones     = list(zone_data.keys())
    gm_status = "Ready" if global_model is not None else "Waiting for 2+ zones"
    gm_color  = "#02C39A" if global_model is not None else "#FF8C00"
    gm_bg     = "#003D2B" if global_model is not None else "#3D1A00"

    labels    = [r.get("timestamp", "")[-8:] for r in readings]
    ph_vals   = [r.get("ph")        or 0 for r in readings]
    tds_vals  = [r.get("tds")       or 0 for r in readings]
    turb_vals = [r.get("turbidity") or 0 for r in readings]
    wqi_vals  = [r.get("wqi")       or 0 for r in readings]
    anomalies = [r.get("anomaly")       for r in readings]

    table_rows = ""
    for r in reversed(readings[-50:]):
        is_anomaly = r.get("anomaly") == "ANOMALY"
        row_style  = 'style="background:#3D0000;color:#FF6B6B;"' if is_anomaly else ''
        badge      = '<span style="background:#FF4444;padding:2px 10px;border-radius:10px;font-size:11px;color:#fff;">ANOMALY</span>' if is_anomaly else '<span style="background:#00A86B;padding:2px 10px;border-radius:10px;font-size:11px;color:#fff;">NORMAL</span>'
        table_rows += f"""
        <tr {row_style}>
            <td>{r.get('received_at','')}</td>
            <td>{r.get('zone','')}</td>
            <td>{r.get('ph','')}</td>
            <td>{r.get('tds','')}</td>
            <td>{r.get('turbidity','')}</td>
            <td>{r.get('wqi','')}</td>
            <td>{badge}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Water Quality Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0A0F1E; color:#E0E6F0; font-family:Arial,sans-serif; padding:20px; }}
  h1 {{ text-align:center; color:#02C39A; font-size:24px; margin-bottom:6px; letter-spacing:1px; }}
  .subtitle {{ text-align:center; color:#64748B; font-size:12px; margin-bottom:24px; }}
  .cards {{ display:flex; gap:15px; margin-bottom:20px; flex-wrap:wrap; }}
  .card {{ flex:1; min-width:150px; background:#161B2E; border-radius:10px; padding:18px; text-align:center; border:1px solid #1E293B; }}
  .card .num {{ font-size:38px; font-weight:bold; }}
  .card .lbl {{ font-size:12px; color:#64748B; margin-top:5px; }}
  .card.blue   .num {{ color:#4FC3F7; }}
  .card.green  .num {{ color:#02C39A; }}
  .card.red    .num {{ color:#FF4444; }}
  .card.yellow .num {{ color:#FFD700; }}
  .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:24px; }}
  .chart-box {{ background:#161B2E; border-radius:10px; padding:16px; border:1px solid #1E293B; }}
  .chart-title {{ color:#4FC3F7; font-size:13px; font-weight:bold; margin-bottom:10px; }}
  .gm-row {{ background:#161B2E; border-radius:10px; padding:16px; margin-bottom:20px; border:1px solid #1E293B; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; }}
  .gm-label {{ font-size:14px; font-weight:bold; color:#4FC3F7; margin-bottom:4px; }}
  .gm-zones {{ font-size:12px; color:#64748B; }}
  .gm-badge {{ padding:6px 18px; border-radius:20px; font-size:13px; font-weight:bold; background:{gm_bg}; color:{gm_color}; border:1px solid {gm_color}; }}
  .section-title {{ color:#4FC3F7; font-size:15px; font-weight:bold; margin-bottom:10px; }}
  table {{ width:100%; border-collapse:collapse; background:#161B2E; border-radius:10px; overflow:hidden; }}
  th {{ background:#1C7293; padding:10px 12px; font-size:12px; text-align:left; color:#fff; }}
  td {{ padding:8px 12px; font-size:12px; border-bottom:1px solid #1E293B; }}
  @media(max-width:600px) {{ .charts {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<h1>Water Quality Monitoring Dashboard</h1>
<p class="subtitle">Critical Water Infrastructure Anomaly Detection | Batch 20 | Auto-refreshes every 30 seconds</p>

<div class="cards">
  <div class="card blue"><div class="num">{n_total}</div><div class="lbl">Total Readings</div></div>
  <div class="card green"><div class="num">{n_normal}</div><div class="lbl">Normal</div></div>
  <div class="card red"><div class="num">{n_anomaly}</div><div class="lbl">Anomalies Detected</div></div>
  <div class="card yellow"><div class="num">{len(zones)}</div><div class="lbl">Zones Connected</div></div>
</div>

<div class="gm-row">
  <div>
    <div class="gm-label">Global Model - Federated Learning</div>
    <div class="gm-zones">Zones: {', '.join(zones) if zones else 'None connected yet'}</div>
  </div>
  <div class="gm-badge">{gm_status}</div>
</div>

<div class="charts">
  <div class="chart-box"><div class="chart-title">pH Over Time</div><canvas id="phChart" height="140"></canvas></div>
  <div class="chart-box"><div class="chart-title">WQI Over Time</div><canvas id="wqiChart" height="140"></canvas></div>
  <div class="chart-box"><div class="chart-title">TDS Over Time</div><canvas id="tdsChart" height="140"></canvas></div>
  <div class="chart-box"><div class="chart-title">Turbidity (NTU) Over Time</div><canvas id="turbChart" height="140"></canvas></div>
</div>

<div class="section-title">Recent Readings (Last 50)</div>
<table>
  <thead>
    <tr><th>Time</th><th>Zone</th><th>pH</th><th>TDS</th><th>NTU</th><th>WQI</th><th>Status</th></tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<script>
const labels    = {labels};
const phVals    = {ph_vals};
const tdsVals   = {tds_vals};
const turbVals  = {turb_vals};
const wqiVals   = {wqi_vals};
const anomalies = {anomalies};
const pointColors = anomalies.map(a => a === 'ANOMALY' ? '#FF4444' : '#02C39A');

function makeChart(id, label, data, color) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [{{
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: color + '22',
        pointBackgroundColor: pointColors,
        pointRadius: 4,
        tension: 0.3,
        fill: true
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ labels: {{ color:'#E0E6F0', font:{{ size:11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color:'#64748B', maxTicksLimit:8, font:{{ size:9 }} }}, grid:{{ color:'#1E293B' }} }},
        y: {{ ticks: {{ color:'#64748B', font:{{ size:10 }} }}, grid:{{ color:'#1E293B' }} }}
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


# -- Federated Averaging --
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
