"""
CLOUD SERVER WITH DASHBOARD - Runs on Railway
==============================================
- Cumulative severity counts
- Zone Status panel with active layers
- Recovery log
- All existing panels retained
"""

from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime, timezone, timedelta
from collections import Counter

app = Flask(__name__)

all_readings      = []
zone_data         = {}
zone_latest       = {}   # latest reading per zone
zone_recovery_log = []   # recovery history
global_model      = None
global_scaler     = StandardScaler()

IST = timezone(timedelta(hours=5, minutes=30))

# ─────────────────────────────────────────────
# HIERARCHY
# ─────────────────────────────────────────────

def get_active_layers(severity):
    """
    Two parallel detection tracks:

    Behavioural track (Isolation Forest):
      ANOMALY  → [ANOMALY]
      CAUTION  → [ANOMALY, CAUTION]

    Regulatory track (WQI):
      WARNING  → [WARNING]

    Both tracks merged:
      CRITICAL → [ANOMALY, CAUTION, WARNING, CRITICAL]

    WARNING does NOT include ANOMALY/CAUTION — when WARNING fires
    (flag=1, pred=1) the Isolation Forest sees normal behaviour.
    CRITICAL includes all because both tracks fired simultaneously.
    """
    if severity == "NORMAL":
        return []
    elif severity == "ANOMALY":
        return ["ANOMALY"]
    elif severity == "CAUTION":
        return ["ANOMALY", "CAUTION"]
    elif severity == "WARNING":
        return ["WARNING"]
    elif severity == "CRITICAL":
        return ["ANOMALY", "CAUTION", "WARNING", "CRITICAL"]
    return [severity]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/send-data', methods=['POST'])
def receive_data():
    global global_model
    data = request.get_json()
    zone = data.get("zone", "Unknown")

    severity      = data.get("severity", "NORMAL")
    active_layers = data.get("active_layers", get_active_layers(severity))
    recovered     = data.get("recovered", False)

    reading = {
        "zone":              zone,
        "timestamp":         data.get("timestamp"),
        "ph":                data.get("ph"),
        "tds":               data.get("tds"),
        "turbidity":         data.get("turbidity") or data.get("ntu"),
        "wqi":               data.get("wqi"),
        "wqi_flag":          data.get("wqi_flag", 0),
        "prediction":        data.get("prediction", 1),
        "anomaly_score":     data.get("anomaly_score", 0),
        "streak":            data.get("streak", 0),
        "severity":          severity,
        "active_layers":     active_layers,
        "anomaly":           data.get("anomaly", "NORMAL"),
        "recovered":         recovered,
        "previous_severity": data.get("previous_severity", ""),
        "action":            data.get("action", ""),
        "received_at":       datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    }

    all_readings.append(reading)

    # Track latest per zone
    prev = zone_latest.get(zone, {})
    zone_latest[zone] = reading

    # Recovery log
    if recovered and prev.get("severity", "NORMAL") not in ("NORMAL", ""):
        zone_recovery_log.append({
            "zone":              zone,
            "recovered_at":      reading["received_at"],
            "previous_severity": data.get("previous_severity", prev.get("severity", "")),
            "action":            data.get("action", "NO_ACTION")
        })

    # Zone sensor data for global model
    if zone not in zone_data:
        zone_data[zone] = []
    zone_data[zone].append([
        data.get("ph", 0),
        data.get("tds", 0),
        data.get("turbidity") or data.get("ntu") or 0,
        data.get("wqi", 0)
    ])

    print(f"[{reading['received_at']}] {zone} | Severity={severity} | Layers={active_layers} | Score={reading['anomaly_score']}")

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
        return jsonify({"status": "not ready", "message": f"Need at least 2 zones. Currently: {list(zone_data.keys())}"}), 200
    return jsonify({"status": "ready", "zones_used": list(zone_data.keys()), "total_readings": len(all_readings)}), 200


@app.route('/clear', methods=['GET'])
def clear_data():
    global all_readings, zone_data, zone_latest, zone_recovery_log, global_model
    all_readings      = []
    zone_data         = {}
    zone_latest       = {}
    zone_recovery_log = []
    global_model      = None
    return jsonify({"status": "success", "message": "All data cleared."}), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "project":   "Critical Water Infrastructure Anomaly Detection",
        "status":    "Cloud server running",
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
    readings = all_readings
    n_total  = len(readings)
    zones    = list(zone_data.keys())

    gm_status = "Ready" if global_model is not None else "Waiting for 2+ zones"
    gm_color  = "#02C39A" if global_model is not None else "#FF8C00"
    gm_bg     = "#003D2B" if global_model is not None else "#3D1A00"

    last_updated = datetime.now(IST).strftime("%d %b %Y  %H:%M:%S IST")

    # ── RAW COUNTS ───────────────────────────────────────
    raw_counts = Counter(r.get("severity", "NORMAL") for r in readings
                         if not r.get("recovered", False))

    # ── CORRECTED CUMULATIVE COUNTS ──────────────────────
    # Behavioural track: ANOMALY → CAUTION → (part of) CRITICAL
    # Regulatory track:  WARNING → (part of) CRITICAL
    # These are parallel — WARNING does NOT include ANOMALY/CAUTION
    sev_counts = dict(raw_counts)

    # Behavioural track cumulative
    sev_counts["ANOMALY"] = (raw_counts.get("ANOMALY",  0) +
                              raw_counts.get("CAUTION",  0) +
                              raw_counts.get("CRITICAL", 0))

    sev_counts["CAUTION"] = (raw_counts.get("CAUTION",  0) +
                              raw_counts.get("CRITICAL", 0))

    # Regulatory track cumulative
    sev_counts["WARNING"] = (raw_counts.get("WARNING",  0) +
                              raw_counts.get("CRITICAL", 0))

    # Critical is exact count only
    sev_counts["CRITICAL"] = raw_counts.get("CRITICAL", 0)

    advisory_count = sev_counts.get("ANOMALY", 0)

    total_nz     = max(n_total, 1)
    pct_normal   = round(raw_counts.get("NORMAL",   0) / total_nz * 100)
    pct_anomaly  = round(sev_counts.get("ANOMALY",  0) / total_nz * 100)
    pct_caution  = round(sev_counts.get("CAUTION",  0) / total_nz * 100)
    pct_warning  = round(sev_counts.get("WARNING",  0) / total_nz * 100)
    pct_critical = round(sev_counts.get("CRITICAL", 0) / total_nz * 100)

    has_caution  = raw_counts.get("CAUTION",  0) > 0
    has_critical = raw_counts.get("CRITICAL", 0) > 0

    # ── ALERT PANEL DATA ─────────────────────────────────
    non_recovered = [r for r in readings if not r.get("recovered", False)]
    wqi_alerts      = [r for r in non_recovered if r.get("wqi_flag") == 1][-10:]
    behav_alerts    = [r for r in non_recovered if r.get("prediction") == -1][-10:]
    critical_alerts = [r for r in non_recovered if r.get("severity") == "CRITICAL"][-10:]
    caution_alerts  = [r for r in non_recovered if r.get("severity") == "CAUTION"][-10:]

    # ── CHART DATA ───────────────────────────────────────
    labels     = [r.get("timestamp", "")[-8:] for r in readings]
    ph_vals    = [r.get("ph")        or 0 for r in readings]
    tds_vals   = [r.get("tds")       or 0 for r in readings]
    turb_vals  = [r.get("turbidity") or 0 for r in readings]
    wqi_vals   = [r.get("wqi")       or 0 for r in readings]
    severities = [r.get("severity", "NORMAL") for r in readings]

    # ── TABLE ROWS ───────────────────────────────────────
    badge_map = {
        "NORMAL":   '<span style="background:#006644;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">NORMAL</span>',
        "ANOMALY":  '<span style="background:#0D47A1;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">ANOMALY</span>',
        "CAUTION":  '<span style="background:#BF360C;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">CAUTION</span>',
        "WARNING":  '<span style="background:#F57F17;padding:2px 9px;border-radius:8px;font-size:10px;color:#000;font-weight:bold;">WARNING</span>',
        "CRITICAL": '<span style="background:#B71C1C;padding:2px 9px;border-radius:8px;font-size:10px;color:#fff;font-weight:bold;">CRITICAL</span>'
    }
    row_bg_map = {
        "NORMAL":   "",
        "ANOMALY":  'style="background:#EFF6FF;"',
        "CAUTION":  'style="background:#FFF3E0;"',
        "WARNING":  'style="background:#FFFDE7;"',
        "CRITICAL": 'style="background:#FFEBEE;"'
    }

    table_rows = ""
    for r in reversed(readings[-50:]):
        sev         = r.get("severity", "NORMAL")
        sev_display = "ANOMALY" if sev == "WATCH" else sev
        layers_str  = " + ".join(r.get("active_layers", get_active_layers(sev))) or "—"
        table_rows += f"""
        <tr {row_bg_map.get(sev_display, "")}>
            <td>{r.get('received_at','')}</td>
            <td>{r.get('zone','')}</td>
            <td>{r.get('ph','')}</td>
            <td>{r.get('tds','')}</td>
            <td>{r.get('turbidity','')}</td>
            <td>{r.get('wqi','')}</td>
            <td>{r.get('wqi_flag','')}</td>
            <td>{r.get('anomaly_score','')}</td>
            <td>{r.get('streak','')}</td>
            <td>{layers_str}</td>
            <td>{badge_map.get(sev_display, badge_map.get(sev,''))}</td>
        </tr>"""

    # ── ZONE STATUS CARDS ────────────────────────────────
    sev_styles = {
        "NORMAL":   ("✅", "#02C39A", "#F0FFF4", "No action needed"),
        "ANOMALY":  ("🔵", "#4FC3F7", "#EFF6FF", "Monitoring — no intervention needed"),
        "CAUTION":  ("🟠", "#FF6D00", "#FFF5EE", "Monitor closely"),
        "WARNING":  ("🟡", "#FFD700", "#FFFDE7", "Investigation required"),
        "CRITICAL": ("🔴", "#FF1744", "#FFF0F0", "Immediate action required"),
    }
    recovered_styles = {
        "CRITICAL": ("⚠️", "#FF8C00", "Post-event inspection recommended"),
        "WARNING":  ("✅", "#02C39A", "No action needed"),
        "CAUTION":  ("✅", "#02C39A", "No action needed"),
        "ANOMALY":  ("✅", "#02C39A", "No action needed"),
    }

    zone_status_cards = ""
    if zone_latest:
        for z, r in zone_latest.items():
            sev    = r.get("severity", "NORMAL")
            layers = r.get("active_layers", get_active_layers(sev))
            icon, color, bg, action = sev_styles.get(sev, sev_styles["NORMAL"])
            layers_html = "".join(
                f'<span style="background:{color}22;color:{color};border:1px solid {color};'
                f'border-radius:6px;padding:2px 8px;font-size:11px;margin:2px;display:inline-block;">'
                f'{l}</span>' for l in layers
            ) if layers else '<span style="color:#888;font-size:11px;">None</span>'

            zone_status_cards += f"""
            <div style="flex:1;min-width:220px;background:{bg};border:2px solid {color};
                        border-radius:14px;padding:18px;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
              <div style="font-size:20px;font-weight:bold;color:{color};margin-bottom:6px;">
                {icon} {z}
              </div>
              <div style="font-size:13px;font-weight:bold;color:{color};margin-bottom:10px;">
                Current: {sev}
              </div>
              <div style="font-size:11px;color:#555;margin-bottom:6px;font-weight:600;">
                ACTIVE LAYERS
              </div>
              <div style="margin-bottom:10px;">{layers_html}</div>
              <div style="font-size:12px;background:{"rgba(255,23,68,0.08)" if sev=="CRITICAL" else "rgba(0,0,0,0.04)"};
                          border-radius:8px;padding:8px 10px;color:{color};font-weight:bold;">
                Action: {action}
              </div>
            </div>"""
    else:
        zone_status_cards = '<p style="color:#888;text-align:center;padding:20px;">No zones connected yet.</p>'

    # ── RECOVERY LOG ROWS ────────────────────────────────
    recovery_rows_html = ""
    if zone_recovery_log:
        for rec in reversed(zone_recovery_log[-10:]):
            prev_sev = rec.get("previous_severity", "")
            action   = rec.get("action", "NO_ACTION")
            if action == "POST_INSPECTION":
                action_html = '<span style="color:#FF8C00;font-weight:bold;">⚠️ Post-event inspection recommended</span>'
            else:
                action_html = '<span style="color:#02C39A;font-weight:bold;">✅ No action needed</span>'
            recovery_rows_html += f"""
            <tr>
                <td>{rec.get('recovered_at','')}</td>
                <td>{rec.get('zone','')}</td>
                <td><span style="background:#B71C1C;color:#fff;padding:2px 8px;
                           border-radius:6px;font-size:10px;">{prev_sev}</span></td>
                <td>{action_html}</td>
            </tr>"""
    else:
        recovery_rows_html = '<tr><td colspan="4" style="color:#888;text-align:center;padding:12px;">No recovery events yet</td></tr>'

    # ── ALERT PANEL HELPERS ──────────────────────────────
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

    # ─────────────────────────────────────────────────────
    # HTML
    # ─────────────────────────────────────────────────────

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

  @keyframes pulse-border {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(255,23,68,0.7); border-color:#FF1744; }}
    50%      {{ box-shadow:0 0 0 12px rgba(255,23,68,0); border-color:#FF6B6B; }}
  }}
  @keyframes blink-text {{
    0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }}
  }}
  @keyframes shake {{
    0%,100% {{ transform:translateX(0); }}
    20% {{ transform:translateX(-4px); }}
    40% {{ transform:translateX(4px); }}
    60% {{ transform:translateX(-3px); }}
    80% {{ transform:translateX(3px); }}
  }}
  @keyframes fadeInDown {{ from{{opacity:0;transform:translateY(-20px);}} to{{opacity:1;transform:translateY(0);}} }}
  @keyframes fadeInUp   {{ from{{opacity:0;transform:translateY(20px);}}  to{{opacity:1;transform:translateY(0);}} }}
  @keyframes fadeInLeft {{ from{{opacity:0;transform:translateX(-20px);}} to{{opacity:1;transform:translateX(0);}} }}
  @keyframes countUp    {{ from{{opacity:0;transform:scale(0.5);}}        to{{opacity:1;transform:scale(1);}}     }}
  @keyframes slideInRow {{ from{{opacity:0;transform:translateX(-10px);}} to{{opacity:1;transform:translateX(0);}} }}
  @keyframes pulse-card {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(255,23,68,0.6); }}
    50%      {{ box-shadow:0 0 16px 4px rgba(255,23,68,0.25); }}
  }}
  @keyframes progressFill {{ from{{width:0%;}} to{{width:var(--pct);}} }}

  .banner-critical{{
    background:#FFF0F0;border:2px solid #FF1744;border-radius:10px;
    padding:16px 20px;margin-bottom:14px;
    display:{"flex" if has_critical else "none"};
    align-items:center;gap:16px;
    animation:pulse-border 1.4s ease-in-out infinite, shake 0.6s ease-in-out 0s 3;
  }}
  .banner-critical .banner-title{{font-size:16px;font-weight:bold;color:#FF1744;animation:blink-text 1s ease-in-out infinite;}}
  .banner-critical .ring{{width:48px;height:48px;border-radius:50%;border:3px solid #FF1744;display:flex;align-items:center;justify-content:center;animation:pulse-border 1s ease-in-out infinite;flex-shrink:0;}}

  .banner-caution{{background:#FFF5EE;border:2px solid #FF6D00;border-radius:10px;padding:16px 20px;margin-bottom:14px;display:{"flex" if has_caution else "none"};align-items:center;gap:16px;}}
  .banner-text .banner-sub{{font-size:13px;color:#666;margin-top:5px;}}

  .cards{{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;}}
  .card{{flex:1;min-width:140px;background:#FFFFFF;border-radius:14px;padding:22px 16px;text-align:center;border:1px solid #D1D9E0;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}
  .card .num{{font-size:44px;font-weight:bold;line-height:1.1;animation:countUp 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.3s both;}}
  .card .lbl{{font-size:13px;color:#555E6D;margin-top:6px;font-weight:500;}}

  .sev-row{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;}}
  .sev-item{{flex:1;min-width:130px;border-radius:14px;padding:18px 12px;text-align:center;border:1px solid;box-shadow:0 2px 10px rgba(0,0,0,0.07);}}
  .sev-item .sev-icon{{margin-bottom:8px;}}
  .sev-item .sev-num{{font-size:38px;font-weight:bold;line-height:1.1;animation:countUp 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.4s both;}}
  .sev-item .sev-lbl{{font-size:13px;margin-top:4px;font-weight:600;letter-spacing:0.3px;}}
  .sev-critical-active{{animation:pulse-card 1.4s ease-in-out infinite;}}

  .prog-bar-wrap{{background:#E8ECF0;border-radius:20px;height:8px;overflow:hidden;margin-top:6px;}}
  .prog-bar{{height:8px;border-radius:20px;width:var(--pct);animation:progressFill 1s cubic-bezier(0.4,0,0.2,1) 0.6s both;}}

  .gm-row{{background:#FFFFFF;border-radius:14px;padding:18px;box-shadow:0 2px 10px rgba(0,0,0,0.06);margin-bottom:16px;border:1px solid #D1D9E0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}}
  .gm-label{{font-size:15px;font-weight:bold;color:#1565C0;margin-bottom:4px;}}
  .gm-zones{{font-size:11px;color:#555E6D;}}
  .gm-badge{{padding:8px 20px;border-radius:20px;font-size:13px;font-weight:bold;background:{gm_bg};color:{gm_color};border:1px solid {gm_color};}}

  .section-title{{color:#1565C0;font-size:14px;font-weight:bold;margin:20px 0 10px 0;text-transform:uppercase;letter-spacing:1px;}}

  .alert-panels{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;}}
  .alert-box{{background:#FFFFFF;border-radius:14px;padding:18px;border:1px solid;box-shadow:0 2px 10px rgba(0,0,0,0.06);}}
  .alert-box h3{{font-size:12px;font-weight:bold;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.8px;}}
  .alert-box table{{width:100%;border-collapse:collapse;}}
  .alert-box td,.alert-box th{{padding:7px 8px;font-size:11px;border-bottom:1px solid #E2E8F0;}}

  .zone-status-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;}}

  .recovery-box{{background:#FFFFFF;border-radius:14px;padding:18px;border:1px solid #02C39A;box-shadow:0 2px 10px rgba(0,0,0,0.06);margin-bottom:16px;}}
  .recovery-box h3{{font-size:12px;font-weight:bold;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.8px;color:#02C39A;}}
  .recovery-box table{{width:100%;border-collapse:collapse;}}
  .recovery-box td,.recovery-box th{{padding:7px 8px;font-size:11px;border-bottom:1px solid #E2E8F0;}}

  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;}}
  .chart-box{{background:#FFFFFF;border-radius:14px;padding:18px;border:1px solid #D1D9E0;box-shadow:0 2px 10px rgba(0,0,0,0.06);}}
  .chart-title{{color:#1565C0;font-size:14px;font-weight:700;margin-bottom:12px;}}

  table.main{{width:100%;border-collapse:collapse;background:#FFFFFF;border-radius:10px;overflow:hidden;}}
  table.main th{{background:#1C7293;padding:11px 14px;font-size:12px;text-align:left;color:#fff;font-weight:600;}}
  table.main td{{padding:9px 14px;font-size:12px;border-bottom:1px solid #E2E8F0;}}
  table.main tbody tr{{animation:slideInRow 0.3s ease both;}}

  .live-dot{{display:inline-block;width:8px;height:8px;background:#02C39A;border-radius:50%;margin-right:6px;vertical-align:middle;animation:pulse-card 1.5s ease-in-out infinite;}}
  .last-updated{{text-align:right;font-size:11px;color:#888;margin-bottom:14px;}}

  h1              {{ animation:fadeInDown 0.6s ease both; }}
  .cards          {{ animation:fadeInUp   0.5s ease 0.1s both; }}
  .sev-row        {{ animation:fadeInUp   0.5s ease 0.2s both; }}
  .gm-row         {{ animation:fadeInUp   0.5s ease 0.3s both; }}
  .alert-panels   {{ animation:fadeInUp   0.5s ease 0.4s both; }}
  .charts         {{ animation:fadeInUp   0.5s ease 0.5s both; }}
  table.main      {{ animation:fadeInUp   0.5s ease 0.6s both; }}

  @media(max-width:700px){{.alert-panels{{grid-template-columns:1fr;}}.charts{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>

<h1>Water Quality Monitoring Dashboard</h1>
<div class="last-updated"><span class="live-dot"></span>Last updated: {last_updated} &nbsp;|&nbsp; Auto-refreshes every 30 seconds</div>

<!-- CRITICAL BANNER -->
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
  <div class="card" style="border-color:#1E293B;"><div class="num" style="color:#02C39A;">{raw_counts.get("NORMAL",0)}</div><div class="lbl">Normal</div></div>
</div>

<!-- SEVERITY OVERVIEW -->
<div class="section-title">Severity Overview</div>
<div class="sev-row">

  <div class="sev-item" style="border-color:#02C39A;background:#F0FFF4;">
    <div class="sev-icon"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#02C39A" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div>
    <div class="sev-num" style="color:#02C39A;">{raw_counts.get("NORMAL",0)}</div>
    <div class="sev-lbl" style="color:#02C39A;">Normal</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#02C39A;--pct:{pct_normal}%;"></div></div>
  </div>

  <div class="sev-item" style="border-color:#4FC3F7;background:#EFF6FF;">
    <div class="sev-icon"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#4FC3F7" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="#4FC3F7"/></svg></div>
    <div class="sev-num" style="color:#4FC3F7;">{advisory_count}</div>
    <div class="sev-lbl" style="color:#4FC3F7;">Advisory (cumulative)</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#4FC3F7;--pct:{pct_anomaly}%;"></div></div>
  </div>

  <div class="sev-item" style="border-color:#FF6D00;background:#FFF5EE;">
    <div class="sev-icon"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FF6D00" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="#FF6D00"/></svg></div>
    <div class="sev-num" style="color:#FF6D00;">{sev_counts.get("CAUTION",0)}</div>
    <div class="sev-lbl" style="color:#FF6D00;">Caution (cumulative)</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#FF6D00;--pct:{pct_caution}%;"></div></div>
  </div>

  <div class="sev-item" style="border-color:#FFD700;background:#FFFDE7;">
    <div class="sev-icon"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FFD700" stroke-width="2"><polygon points="12 2 22 20 2 20"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="1" fill="#FFD700"/></svg></div>
    <div class="sev-num" style="color:#FFD700;">{sev_counts.get("WARNING",0)}</div>
    <div class="sev-lbl" style="color:#FFD700;">Warning (cumulative)</div>
    <div class="prog-bar-wrap"><div class="prog-bar" style="background:#FFD700;--pct:{pct_warning}%;"></div></div>
  </div>

  <div class="sev-item {"sev-critical-active" if has_critical else ""}" style="border-color:#FF1744;background:#FFF0F0;">
    <div class="sev-icon"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FF1744" stroke-width="2.5"><polygon points="12 2 22 20 2 20"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="1" fill="#FF1744"/></svg></div>
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

<!-- ALERT PANELS (existing — unchanged) -->
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

  <div class="alert-box" style="border-color:#FF1744;">
    <h3 style="color:#FF1744;">Critical &mdash; Confirmed Contamination Event</h3>
    <table>
      <tr><th style="color:#FF1744;">Time</th><th style="color:#FF1744;">Zone</th><th style="color:#FF1744;">pH</th><th style="color:#FF1744;">TDS</th><th style="color:#FF1744;">WQI</th><th style="color:#FF1744;">Score</th></tr>
      {critical_rows(critical_alerts)}
    </table>
  </div>

</div>

<!-- CAUTION PANEL (existing — unchanged) -->
<div class="section-title">Sustained Behavioural Deviation &mdash; Caution Events</div>
<div class="alert-box" style="border-color:#FF6D00;margin-bottom:16px;">
  <h3 style="color:#FF6D00;">Consecutive Anomaly Streak &mdash; 3 or More Readings</h3>
  <table style="width:100%">
    <tr><th style="color:#FF6D00;">Time</th><th style="color:#FF6D00;">Zone</th><th style="color:#FF6D00;">Score</th><th style="color:#FF6D00;">Streak</th></tr>
    {caution_rows(caution_alerts)}
  </table>
</div>

<!-- ★ NEW — ZONE STATUS PANEL -->
<div class="section-title">Zone Status &amp; Action Required</div>
<div class="zone-status-row">
  {zone_status_cards}
</div>

<!-- ★ NEW — RECOVERY LOG -->
<div class="section-title">Recovery Log</div>
<div class="recovery-box">
  <h3>Past Recovery Events &mdash; Action Recommendation</h3>
  <table>
    <tr>
      <th style="color:#02C39A;">Recovered At</th>
      <th style="color:#02C39A;">Zone</th>
      <th style="color:#02C39A;">Previous Severity</th>
      <th style="color:#02C39A;">Action</th>
    </tr>
    {recovery_rows_html}
  </table>
</div>

<!-- CHARTS (existing — unchanged) -->
<div class="section-title">Sensor Trends</div>
<div class="charts">
  <div class="chart-box"><div class="chart-title">pH Over Time</div><canvas id="phChart"   height="130"></canvas></div>
  <div class="chart-box"><div class="chart-title">WQI Over Time</div><canvas id="wqiChart" height="130"></canvas></div>
  <div class="chart-box"><div class="chart-title">TDS Over Time</div><canvas id="tdsChart" height="130"></canvas></div>
  <div class="chart-box"><div class="chart-title">Turbidity (NTU) Over Time</div><canvas id="turbChart" height="130"></canvas></div>
</div>

<!-- READINGS TABLE (existing + active_layers column added) -->
<div class="section-title">All Readings &mdash; Last 50</div>
<table class="main">
  <thead>
    <tr>
      <th>Timestamp</th><th>Zone</th><th>pH</th><th>TDS</th><th>NTU</th>
      <th>WQI</th><th>WQI Flag</th><th>Score</th><th>Streak</th>
      <th>Active Layers</th><th>Severity</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<script>
const labels   = {labels};
const phVals   = {ph_vals};
const tdsVals  = {tds_vals};
const turbVals = {turb_vals};
const wqiVals  = {wqi_vals};
const sevs     = {severities};
const colorMap = {{NORMAL:'#02C39A',WATCH:'#4FC3F7',ANOMALY:'#4FC3F7',CAUTION:'#FF6D00',WARNING:'#FFD700',CRITICAL:'#FF1744'}};
const ptColors = sevs.map(s => colorMap[s] || '#02C39A');

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
        y: {{ ticks: {{ color:'#64748B', font:{{ size:9 }} }},              grid:{{ color:'#E2E8F0' }} }}
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
    print(f"[CLOUD] Global Model ready — Zones: {list(zone_data.keys())} | Samples: {len(combined)}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
