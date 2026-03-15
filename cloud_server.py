"""
CLOUD SERVER - Runs on Railway
================================
Fixed: Global model activates when 2+ zones send data
"""

from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# ── In-memory storage ──
all_readings  = []
zone_data     = {}   # stores readings per zone
global_model  = None
global_scaler = StandardScaler()


# ─────────────────────────────────────────────
# ROUTE 1: Receive data from laptop
# POST /send-data
# ─────────────────────────────────────────────
@app.route('/send-data', methods=['POST'])
def receive_data():
    global global_model

    data = request.get_json()
    zone = data.get("zone", "Unknown")

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

    # Store readings per zone
    if zone not in zone_data:
        zone_data[zone] = []
    zone_data[zone].append([
        data.get("ph", 0),
        data.get("tds", 0),
        data.get("turbidity") or data.get("ntu") or 0,
        data.get("wqi", 0)
    ])

    print(f"[{reading['received_at']}] {zone} | pH={reading['ph']} "
          f"TDS={reading['tds']} WQI={reading['wqi']} Anomaly={reading['anomaly']}")

    # Build global model when 2+ zones have data
    if len(zone_data) >= 2:
        build_global_model()

    return jsonify({"status": "success", "message": f"Data from {zone} received"}), 200


# ─────────────────────────────────────────────
# ROUTE 2: View all stored data
# GET /data
# ─────────────────────────────────────────────
@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({
        "total_readings":     len(all_readings),
        "zones_connected":    list(zone_data.keys()),
        "global_model_ready": global_model is not None,
        "readings":           all_readings[-20:]
    }), 200


# ─────────────────────────────────────────────
# ROUTE 3: Get global model status
# GET /global-model
# ─────────────────────────────────────────────
@app.route('/global-model', methods=['GET'])
def get_global_model():
    if global_model is None:
        return jsonify({
            "status":  "not ready",
            "message": f"Need at least 2 zones. Currently have: {list(zone_data.keys())}"
        }), 200

    return jsonify({
        "status":         "ready",
        "zones_used":     list(zone_data.keys()),
        "total_readings": len(all_readings),
        "message":        "Global model trained and ready!"
    }), 200


# ─────────────────────────────────────────────
# ROUTE 4: Clear all data
# GET /clear
# ─────────────────────────────────────────────
@app.route('/clear', methods=['GET'])
def clear_data():
    global all_readings, zone_data, global_model
    all_readings = []
    zone_data    = {}
    global_model = None
    print("[CLOUD] All data cleared!")
    return jsonify({
        "status":  "success",
        "message": "All data cleared. Total readings: 0"
    }), 200


# ─────────────────────────────────────────────
# ROUTE 5: Health check
# GET /
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "project": "Critical Water Infrastructure Anomaly Detection",
        "status":  "Cloud server running",
        "endpoints": {
            "POST /send-data":   "Send sensor data from laptop",
            "GET /data":         "View all stored readings",
            "GET /global-model": "Check global model status",
            "GET /clear":        "Clear all stored data"
        }
    }), 200


# ─────────────────────────────────────────────
# FEDERATED AVERAGING
# Combines data from all zones → Global Model
# ─────────────────────────────────────────────
def build_global_model():
    global global_model, global_scaler

    print(f"\n[CLOUD] Building Global Model from {len(zone_data)} zones: {list(zone_data.keys())}")

    # Combine all zone data
    all_zone_data = []
    for zone, readings in zone_data.items():
        all_zone_data.extend(readings)

    combined = np.array(all_zone_data)

    global_scaler = StandardScaler()
    scaled = global_scaler.fit_transform(combined)

    global_model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    global_model.fit(scaled)

    print(f"[CLOUD] Global Model ready! Zones: {list(zone_data.keys())} | Samples: {len(combined)}\n")


# ─────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print(" Water Quality Cloud Server Starting...")
    print(f" Listening on port {port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port)
