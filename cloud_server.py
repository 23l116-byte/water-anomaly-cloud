"""
CLOUD SERVER WITH DASHBOARD - Runs on Railway
"""

from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime, timezone, timedelta

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
        "received_at":   datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
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

    from collections import Counter
    import json as _json
    sev_counts = Counter(r.get("severity", "NORMAL") for r in readings)

    # Summary stats
    ph_list   = [r.get("ph")  or 0 for r in readings if r.get("ph")]
    tds_list  = [r.get("tds") or 0 for r in readings if r.get("tds")]
    wqi_list  = [r.get("wqi") or 0 for r in readings if r.get("wqi")]

    ph_avg  = round(sum(ph_list) /len(ph_list),  2) if ph_list  else 0
    tds_avg = round(sum(tds_list)/len(tds_list), 1) if tds_list else 0
    wqi_avg = round(sum(wqi_list)/len(wqi_list), 1) if wqi_list else 0
    ph_min  = round(min(ph_list),  2) if ph_list  else 0
    ph_max  = round(max(ph_list),  2) if ph_list  else 0
    wqi_min = round(min(wqi_list), 1) if wqi_list else 0
    wqi_max = round(max(wqi_list), 1) if wqi_list else 0

    # Progress bar percentages
    total_nz = max(n_total, 1)
    pct_normal   = round(sev_counts.get("NORMAL",  0)/total_nz*100)
    pct_anomaly  = round(sev_counts.get("ANOMALY", 0)/total_nz*100) + round(sev_counts.get("WATCH",0)/total_nz*100)
    pct_caution  = round(sev_counts.get("CAUTION", 0)/total_nz*100)
    pct_warning  = round(sev_counts.get("WARNING", 0)/total_nz*100)
    pct_critical = round(sev_counts.get("CRITICAL",0)/total_nz*100)

    from datetime import datetime, timezone, timedelta as _dt
    from datetime import timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    last_updated = _dt.now(IST).strftime("%d %b %Y  %H:%M:%S IST")

    wqi_alerts      = [r for r in readings if r.get("wqi_flag") == 1][-10:]
    behav_alerts    = [r for r in readings if r.get("prediction") == -1][-10:]
    critical_alerts = [r for r in readings if r.get("severity") == "CRITICAL"][-10:]
    caution_alerts  = [r for r in readings if r.get("severity") == "CAUTION"][-10:]

    has_caution  = sev_counts.get("CAUTION",  0) > 0
    has_critical = sev_counts.get("CRITICAL", 0) > 0

    labels     = [r.get("timestamp", "")[-8:] for r in readings]
    ph_vals    = [r.get("ph")        or 0 for r in readings]
    tds_vals   = [r.get("tds")       or 0 for r in readings]
    turb_vals  = [r.get("turbidity") or 0 for r in readings]
    wqi_vals   = [r.get("wqi")       or 0 for r in readings]
    severities = [r.get("severity", "NORMAL") for r in readings]

    # WATCH → ANOMALY
    badge_map = {
        "NORMAL":   '<span style="background:#006644;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">NORMAL</span>',
        "ANOMALY": '<span style="background:#0D47A1;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">ANOMALY</span>',
        "CAUTION":  '<span style="background:#BF360C;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">CAUTION</span>',
        "WARNING":  '<span style="background:#F57F17;padding:2px 9px;border-radius:8px;font-size:10px;color:#000;font-weight:bold;">WARNING</span>',
        "CRITICAL": '<span style="background:#B71C1C;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">CRITICAL</span>'
    }
    row_bg_map = {
        "NORMAL":   "",
        "ANOMALY": 'style="background:#EFF6FF;color:#90CAF9;"',
        "CAUTION":  'style="background:#FFF3E0;color:#FFAB76;"',
        "WARNING":  'style="background:#FFFDE7;color:#FFE082;"',
        "CRITICAL": 'style="background:#FFEBEE;color:#FF8A80;"'
    }

    table_rows = ""
    for r in reversed(readings[-50:]):
        sev = r.get("severity", "NORMAL")
        # map WATCH to ANOMALY in display
        sev_display = "ANOMALY" if sev == "WATCH" else sev
        table_rows += f"""
        <tr {row_bg_map.get(sev_display, row_bg_map.get(sev, ""))}>
            <td>{r.get('received_at','')}</td>
            <td>{r.get('zone','')}</td>
            <td>{r.get('ph','')}</td>
            <td>{r.get('tds','')}</td>
            <td>{r.get('turbidity','')}</td>
            <td>{r.get('wqi','')}</td>
            <td>{r.get('wqi_flag','')}</td>
            <td>{r.get('anomaly_score','')}</td>
            <td>{r.get('streak','')}</td>
            <td>{badge_map.get(sev_display, badge_map.get(sev,''))}</td>
        </tr>"""

    def wqi_rows(alerts):
        if not alerts:
            return "<tr><td colspan='3' style='color:#555;text-align:center;padding:10px;'>No regulatory alerts</td></tr>"
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td style='color:#FF8C00;font-weight:bold;'>{r.get('wqi','')}</td></tr>"
        return rows

    def behav_rows(alerts):
        if not alerts:
            return "<tr><td colspan='4' style='color:#555;text-align:center;padding:10px;'>No behavioural alerts</td></tr>"
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('anomaly_score','')}</td><td>{r.get('streak','')}</td></tr>"
        return rows

    def critical_rows(alerts):
        if not alerts:
            return "<tr><td colspan='6' style='color:#555;text-align:center;padding:10px;'>No critical alerts</td></tr>"
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('ph','')}</td><td>{r.get('tds','')}</td><td>{r.get('wqi','')}</td><td>{r.get('anomaly_score','')}</td></tr>"
        return rows

    def caution_rows(alerts):
        if not alerts:
            return "<tr><td colspan='4' style='color:#555;text-align:center;padding:10px;'>No caution alerts</td></tr>"
        rows = ""
        for r in reversed(alerts):
            rows += f"<tr><td>{r.get('received_at','')}</td><td>{r.get('zone','')}</td><td>{r.get('anomaly_score','')}</td><td style='color:#FF8C00;font-weight:bold;'>{r.get('streak','')}</td></tr>"
        return rows

    advisory_count = sev_counts.get("WATCH", 0) + sev_counts.get("ANOMALY", 0)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Water Quality Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#F0F4F8;color:#1A1A2E;font-family:Arial,sans-serif;padding:28px 32px;}}
  h1{{text-align:center;color:#02C39A;font-size:28px;margin-bottom:8px;letter-spacing:1.5px;font-weight:800;}}
  .subtitle{{text-align:center;color:#555E6D;font-size:11px;margin-bottom:20px;}}

  /* ── CRITICAL ANIMATED BANNER ── */
  @keyframes pulse-border {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(255,23,68,0.7); border-color:#FF1744; }}
    50%      {{ box-shadow: 0 0 0 12px rgba(255,23,68,0); border-color:#FF6B6B; }}
  }}
  @keyframes blink-text {{
    0%,100% {{ opacity:1; }}
    50%      {{ opacity:0.4; }}
  }}
  @keyframes shake {{
    0%,100% {{ transform:translateX(0); }}
    20%     {{ transform:translateX(-4px); }}
    40%     {{ transform:translateX(4px); }}
    60%     {{ transform:translateX(-3px); }}
    80%     {{ transform:translateX(3px); }}
  }}
  @keyframes spin-icon {{
    0%   {{ transform:rotate(0deg); }}
    100% {{ transform:rotate(360deg); }}
  }}

  .banner-critical{{
    background:#FFF0F0;
    border:2px solid #FF1744;
    border-radius:10px;
    padding:16px 20px;
    margin-bottom:14px;
    display:{"flex" if has_critical else "none"};
    align-items:center;gap:16px;
    animation: pulse-border 1.4s ease-in-out infinite, shake 0.6s ease-in-out 0s 3;
  }}
  .banner-critical .banner-title{{
    font-size:16px;font-weight:bold;
    color:#FF1744;
    animation: blink-text 1s ease-in-out infinite;
  }}
  .banner-critical .ring{{
    width:48px;height:48px;border-radius:50%;
    border:3px solid #FF1744;
    display:flex;align-items:center;justify-content:center;
    animation: pulse-border 1s ease-in-out infinite;
    flex-shrink:0;
  }}

  .banner-caution{{
    background:#FFF5EE;border:2px solid #FF6D00;border-radius:10px;
    padding:16px 20px;margin-bottom:14px;
    display:{"flex" if has_caution else "none"};
    align-items:center;gap:16px;
  }}
  .banner-text .banner-sub{{font-size:13px;color:#666;margin-top:5px;}}

  /* STAT CARDS */
  .cards{{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;}}
  .card{{flex:1;min-width:140px;background:#FFFFFF;border-radius:14px;padding:22px 16px;text-align:center;border:1px solid #D1D9E0;box-shadow:0 2px 12px rgba(0,0,0,0.08);transition:transform 0.2s;}} .card:hover{{transform:translateY(-2px);}}
  .card .num{{font-size:44px;font-weight:bold;line-height:1.1;}}
  .card .lbl{{font-size:13px;color:#555E6D;margin-top:6px;font-weight:500;}}

  /* SEVERITY PICTOGRAPH */
  .sev-row{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;}}
  .sev-item{{flex:1;min-width:130px;border-radius:14px;padding:18px 12px;text-align:center;border:1px solid;box-shadow:0 2px 10px rgba(0,0,0,0.07);transition:transform 0.2s;}} .sev-item:hover{{transform:translateY(-2px);}}
  .sev-item .sev-icon{{margin-bottom:8px;}}
  .sev-item .sev-num{{font-size:38px;font-weight:bold;line-height:1.1;}}
  .sev-item .sev-lbl{{font-size:13px;margin-top:4px;font-weight:600;letter-spacing:0.3px;}}

  /* CRITICAL SEV CARD ANIMATION */
  @keyframes pulse-card {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(255,23,68,0.6); }}
    50%      {{ box-shadow:0 0 16px 4px rgba(255,23,68,0.25); }}
  }}
  .sev-critical-active{{animation: pulse-card 1.4s ease-in-out infinite;}}

  /* GLOBAL MODEL */
  .gm-row{{background:#FFFFFF;border-radius:14px;padding:18px;box-shadow:0 2px 10px rgba(0,0,0,0.06);margin-bottom:16px;border:1px solid #D1D9E0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}}
  .gm-label{{font-size:15px;font-weight:bold;color:#1565C0;margin-bottom:4px;}}
  .gm-zones{{font-size:11px;color:#555E6D;}}
  .gm-badge{{padding:8px 20px;border-radius:20px;font-size:13px;font-weight:bold;background:{gm_bg};color:{gm_color};border:1px solid {gm_color};}}

  /* ALERT PANELS */
  .section-title{{color:#1565C0;font-size:14px;font-weight:bold;margin:20px 0 10px 0;text-transform:uppercase;letter-spacing:1px;}}
  .alert-panels{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;}}
  .alert-box{{background:#FFFFFF;border-radius:14px;padding:18px;border:1px solid;box-shadow:0 2px 10px rgba(0,0,0,0.06);}}
  .alert-box h3{{font-size:12px;font-weight:bold;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.8px;}}
  .alert-box table{{width:100%;border-collapse:collapse;}}
  .alert-box td,.alert-box th{{padding:7px 8px;font-size:11px;border-bottom:1px solid #E2E8F0;}}

  /* CHARTS */
  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;}}
  .chart-box{{background:#FFFFFF;border-radius:14px;padding:18px;border:1px solid #D1D9E0;box-shadow:0 2px 10px rgba(0,0,0,0.06);}}
  .chart-title{{color:#1565C0;font-size:14px;font-weight:700;margin-bottom:12px;}}

  /* TABLE */
  table.main{{width:100%;border-collapse:collapse;background:#FFFFFF;border-radius:10px;overflow:hidden;}}
  table.main th{{background:#1C7293;padding:11px 14px;font-size:12px;text-align:left;color:#fff;font-weight:600;}}
  table.main td{{padding:9px 14px;font-size:12px;border-bottom:1px solid #E2E8F0;}}


  /* ── ANIMATIONS ── */
  @keyframes fadeInDown {{
    from {{ opacity:0; transform:translateY(-20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes fadeInUp {{
    from {{ opacity:0; transform:translateY(20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes fadeInLeft {{
    from {{ opacity:0; transform:translateX(-20px); }}
    to   {{ opacity:1; transform:translateX(0); }}
  }}
  @keyframes countUp {{
    from {{ opacity:0; transform:scale(0.5); }}
    to   {{ opacity:1; transform:scale(1); }}
  }}
  @keyframes slideInRow {{
    from {{ opacity:0; transform:translateX(-10px); }}
    to   {{ opacity:1; transform:translateX(0); }}
  }}
  @keyframes glowGreen {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(2,195,154,0.4); }}
    50%      {{ box-shadow:0 0 16px 4px rgba(2,195,154,0.15); }}
  }}
  @keyframes glowOrange {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(255,109,0,0.4); }}
    50%      {{ box-shadow:0 0 16px 4px rgba(255,109,0,0.2); }}
  }}
  @keyframes progressFill {{
    from {{ width:0%; }}
    to   {{ width:var(--pct); }}
  }}
  @keyframes rotateDot {{
    0%   {{ transform:rotate(0deg)   translateX(18px) rotate(0deg);   }}
    100% {{ transform:rotate(360deg) translateX(18px) rotate(-360deg);}}
  }}

  /* Apply fade-in to major sections */
  h1               {{ animation: fadeInDown 0.6s ease both; }}
  .cards           {{ animation: fadeInUp   0.5s ease 0.1s both; }}
  .sev-row         {{ animation: fadeInUp   0.5s ease 0.2s both; }}
  .gm-row          {{ animation: fadeInUp   0.5s ease 0.3s both; }}
  .alert-panels    {{ animation: fadeInUp   0.5s ease 0.4s both; }}
  .charts          {{ animation: fadeInUp   0.5s ease 0.5s both; }}
  table.main       {{ animation: fadeInUp   0.5s ease 0.6s both; }}

  /* Stat card number pop */
  .card .num       {{ animation: countUp    0.5s cubic-bezier(0.34,1.56,0.64,1) 0.3s both; }}

  /* Severity card numbers */
  .sev-num         {{ animation: countUp    0.5s cubic-bezier(0.34,1.56,0.64,1) 0.4s both; }}

  /* Glow on active severity cards */
  .sev-normal-active  {{ animation: glowGreen  2.5s ease-in-out infinite; }}
  .sev-caution-active {{ animation: glowOrange 2s   ease-in-out infinite; }}

  /* Table row stagger */
  table.main tbody tr {{
    animation: slideInRow 0.3s ease both;
  }}
  table.main tbody tr:nth-child(1)  {{ animation-delay:0.05s; }}
  table.main tbody tr:nth-child(2)  {{ animation-delay:0.10s; }}
  table.main tbody tr:nth-child(3)  {{ animation-delay:0.15s; }}
  table.main tbody tr:nth-child(4)  {{ animation-delay:0.20s; }}
  table.main tbody tr:nth-child(5)  {{ animation-delay:0.25s; }}
  table.main tbody tr:nth-child(6)  {{ animation-delay:0.30s; }}
  table.main tbody tr:nth-child(7)  {{ animation-delay:0.35s; }}
  table.main tbody tr:nth-child(8)  {{ animation-delay:0.40s; }}
  table.main tbody tr:nth-child(9)  {{ animation-delay:0.45s; }}
  table.main tbody tr:nth-child(10) {{ animation-delay:0.50s; }}

  /* Alert panel stagger */
  .alert-box {{
    animation: fadeInLeft 0.4s ease both;
  }}

  /* Progress bar for severity distribution */
  .prog-bar-wrap {{
    background:#E8ECF0; border-radius:20px; height:8px;
    overflow:hidden; margin-top:6px;
  }}
  .prog-bar {{
    height:8px; border-radius:20px;
    width:var(--pct);
    animation: progressFill 1s cubic-bezier(0.4,0,0.2,1) 0.6s both;
  }}

  /* Spinning dot loader for last-updated */
  .live-dot {{
    display:inline-block; width:8px; height:8px;
    background:#02C39A; border-radius:50%;
    margin-right:6px; vertical-align:middle;
    animation: pulse-card 1.5s ease-in-out infinite;
  }}

  /* Stats row */
  .stats-row {{ display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }}
  .stat-mini {{
    flex:1; min-width:100px; background:#FFFFFF;
    border-radius:10px; padding:12px 14px;
    border:1px solid #D1D9E0;
    box-shadow:0 1px 6px rgba(0,0,0,0.05);
    animation: fadeInUp 0.5s ease 0.5s both;
  }}
  .stat-mini .sm-label {{ font-size:11px; color:#888; margin-bottom:4px; }}
  .stat-mini .sm-val   {{ font-size:18px; font-weight:bold; color:#1565C0; }}

  /* Last updated bar */
  .last-updated {{
    text-align:right; font-size:11px; color:#888;
    margin-bottom:14px; animation: fadeInDown 0.5s ease 0.7s both;
  }}
  @media(max-width:700px){{.alert-panels{{grid-template-columns:1fr;}}.charts{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>

<h1>Water Quality Monitoring Dashboard</h1>

<div class="last-updated"><span class="live-dot"></span>Last updated: {last_updated} &nbsp;|&nbsp; Auto-refreshes every 30 seconds</div>

<!-- CRITICAL ANIMATED BANNER -->
<div class="banner-critical">
  <div class="ring">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#FF1744" stroke-width="2.5">
      <polygon points="12 2 22 20 2 20"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <circle cx="12" cy="17" r="0.8" fill="#FF1744"/>
    </svg>
  </div>
  <div class="banner-text">
    <div class="banner-title">IMMEDIATE ATTENTION REQUIRED</div>
    <div class="banner-sub">Critical anomaly confirmed. Both detection layers have triggered with a sustained streak. Investigate affected zones immediately.</div>
  </div>
</div>

<!-- CAUTION BANNER -->
<div class="banner-caution">
  <div style="flex-shrink:0;">
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#FF6D00" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <circle cx="12" cy="16" r="0.8" fill="#FF6D00"/>
    </svg>
  </div>
  <div class="banner-text">
    <div class="banner-title" style="color:#FF6D00;font-size:17px;font-weight:bold;">ATTENTION REQUIRED &mdash; ALERT</div>
    <div class="banner-sub">Sustained behavioural deviation detected across 3 or more consecutive readings. Regulatory limits are currently met. Monitor closely.</div>
  </div>
</div>

<!-- STAT CARDS -->
<div class="cards">
  <div class="card" style="border-color:#4FC3F7;"><div class="num" style="color:#4FC3F7;">{n_total}</div><div class="lbl">Total Readings</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#FFD700;">{len(zones)}</div><div class="lbl">Zones Connected</div></div>
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#02C39A;">{sev_counts.get("NORMAL",0)}</div><div class="lbl">Normal</div></div>
</div>

<!-- SEVERITY PICTOGRAPH ROW -->
<div class="section-title">Severity Overview</div>
<div class="sev-row">

  <div class="sev-item" style="border-color:#02C39A;background:#F0FFF4;">
    <div class="sev-icon">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#02C39A" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
      </svg>
    </div>
    <div class="sev-num" style="color:#02C39A;">{sev_counts.get("NORMAL",0)}</div>
    <div class="sev-lbl" style="color:#02C39A;">Normal</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#02C39A;--pct:{pct_normal}%;"></div></div>
  </div>

  <div class="sev-item" style="border-color:#4FC3F7;background:#EFF6FF;">
    <div class="sev-icon">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#4FC3F7" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <circle cx="12" cy="16" r="1" fill="#4FC3F7"/>
      </svg>
    </div>
    <div class="sev-num" style="color:#4FC3F7;">{advisory_count}</div>
    <div class="sev-lbl" style="color:#4FC3F7;">Advisory</div>
  </div>

  <div class="sev-item" style="border-color:#FF6D00;background:#FFF5EE;{"border-width:2px;" if has_caution else ""}">
    <div class="sev-icon">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FF6D00" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <circle cx="12" cy="16" r="1" fill="#FF6D00"/>
      </svg>
    </div>
    <div class="sev-num" style="color:#FF6D00;">{sev_counts.get("CAUTION",0)}</div>
    <div class="sev-lbl" style="color:#FF6D00;">Caution</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#FF6D00;--pct:{pct_caution}%;"></div></div>
  </div>

  <div class="sev-item" style="border-color:#FFD700;background:#FFFDE7;">
    <div class="sev-icon">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FFD700" stroke-width="2">
        <polygon points="12 2 22 20 2 20"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <circle cx="12" cy="17" r="1" fill="#FFD700"/>
      </svg>
    </div>
    <div class="sev-num" style="color:#FFD700;">{sev_counts.get("WARNING",0)}</div>
    <div class="sev-lbl" style="color:#FFD700;">Warning</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#FFD700;--pct:{pct_warning}%;"></div></div>
  </div>

  <div class="sev-item {"sev-critical-active" if has_critical else ""}" style="border-color:#FF1744;background:#FFF0F0;{"border-width:2px;" if has_critical else ""}">
    <div class="sev-icon">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FF1744" stroke-width="2.5">
        <polygon points="12 2 22 20 2 20"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <circle cx="12" cy="17" r="1" fill="#FF1744"/>
      </svg>
    </div>
    <div class="sev-num" style="color:#FF1744;">{sev_counts.get("CRITICAL",0)}</div>
    <div class="sev-lbl" style="color:#FF1744;">Critical</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#FF1744;--pct:{pct_critical}%;"></div></div>
  </div>

</div>

<!-- GLOBAL MODEL -->
<div class="gm-row">
  <div>
    <div class="gm-label">Global Model &mdash; Federated Learning</div>
    <div class="gm-zones">Zones Connected: {', '.join(zones) if zones else 'None'}</div>
  </div>
  <div class="gm-badge">{gm_status}</div>
</div>

<!-- ALERT PANELS -->
<div class="section-title">Alert Panels</div>
<div class="alert-panels">

  <div class="alert-box" style="border-color:#FF8C00;">
    <h3 style="color:#FF8C00;">Regulatory Breach &mdash; WQI Limit Exceeded</h3>
    <table>
      <tr><th style="color:#FF8C00;">Time</th><th style="color:#FF8C00;">Zone</th><th style="color:#FF8C00;">WQI</th></tr>
      {wqi_rows(wqi_alerts)}
    </table>
  </div>

  <div class="alert-box" style="border-color:#4FC3F7;">
    <h3 style="color:#4FC3F7;">Behavioural Anomaly &mdash; Deviation Detected</h3>
    <table>
      <tr><th style="color:#4FC3F7;">Time</th><th style="color:#4FC3F7;">Zone</th><th style="color:#4FC3F7;">Score</th><th style="color:#4FC3F7;">Streak</th></tr>
      {behav_rows(behav_alerts)}
    </table>
  </div>

  <div class="alert-box" style="border-color:#FF1744;{"animation:pulse-border 1.4s ease-in-out infinite;" if has_critical else ""}">
    <h3 style="color:#FF1744;">Critical &mdash; Confirmed Contamination Event</h3>
    <table>
      <tr><th style="color:#FF1744;">Time</th><th style="color:#FF1744;">Zone</th><th style="color:#FF1744;">pH</th><th style="color:#FF1744;">TDS</th><th style="color:#FF1744;">WQI</th><th style="color:#FF1744;">Score</th></tr>
      {critical_rows(critical_alerts)}
    </table>
  </div>

</div>

<!-- CAUTION PANEL -->
<div class="section-title">Sustained Behavioural Deviation &mdash; Caution Events</div>
<div class="alert-box" style="border-color:#FF6D00;margin-bottom:16px;">
  <h3 style="color:#FF6D00;">Consecutive Anomaly Streak &mdash; 3 or More Readings</h3>
  <table style="width:100%">
    <tr><th style="color:#FF6D00;">Time</th><th style="color:#FF6D00;">Zone</th><th style="color:#FF6D00;">Score</th><th style="color:#FF6D00;">Streak</th></tr>
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

<!-- TABLE -->
<div class="section-title">All Readings &mdash; Last 50</div>
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
const colorMap  = {{NORMAL:'#02C39A',WATCH:'#4FC3F7',ANOMALY:'#4FC3F7',CAUTION:'#FF6D00',WARNING:'#FFD700',CRITICAL:'#FF1744'}};
const ptColors  = sevs.map(s => colorMap[s] || '#02C39A');

function makeChart(id, label, data, color) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [{{
        label: label, data: data,
        borderColor: color, backgroundColor: color + '18',
        pointBackgroundColor: ptColors,
        pointRadius: 4, tension: 0.3, fill: true
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ labels: {{ color:'#1A1A2E', font:{{ size:10 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color:'#64748B', maxTicksLimit:8, font:{{ size:9 }} }}, grid:{{ color:'#E2E8F0' }} }},
        y: {{ ticks: {{ color:'#64748B', font:{{ size:9  }} }}, grid:{{ color:'#E2E8F0' }} }}
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
