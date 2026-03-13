"""
CLOUD SERVER - Runs on Railway
================================
Receives pH, TDS, Turbidity, WQI, Anomaly + model weights from laptop
Stores all data
Builds Global Model using Federated Averaging
"""

from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)

# ── In-memory storage ──
all_readings = []        # Stores all sensor data received
zone_weights = {}        # Stores model weights per zone
global_model = None      # The global Isolation Forest model
global_scaler = StandardScaler()


# ─────────────────────────────────────────────
# ROUTE 1: Receive data from laptop
# POST /send-data
# ─────────────────────────────────────────────
@app.route('/send-data', methods=['POST'])
def receive_data():
    global global_model, zone_weights

    data = request.get_json()

    # Store the reading
    reading = {
        "zone":       data.get("zone"),
        "timestamp":  data.get("timestamp"),
        "ph":         data.get("ph"),
        "tds":        data.get("tds"),
        "turbidity":  data.get("turbidity"),
        "wqi":        data.get("wqi"),
        "anomaly":    data.get("anomaly"),
        "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    all_readings.append(reading)

    # Store model weights from this zone
    zone = data.get("zone")
    weights = data.get("model_weights")
    if weights:
        zone_weights[zone] = np.array(weights)

    print(f"[{reading['received_at']}] Received from {zone} | "
          f"pH={reading['ph']} TDS={reading['tds']} "
          f"Turbidity={reading['turbidity']} WQI={reading['wqi']} "
          f"Anomaly={reading['anomaly']}")

    # Try to build global model if we have weights from at least 2 zones
    if len(zone_weights) >= 2:
        build_global_model()

    return jsonify({"status": "success", "message": f"Data from {zone} received"}), 200


# ─────────────────────────────────────────────
# ROUTE 2: View all stored data
# GET /data
# ─────────────────────────────────────────────
@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({
        "total_readings": len(all_readings),
        "zones_connected": list(zone_weights.keys()),
        "global_model_ready": global_model is not None,
        "readings": all_readings[-20:]  # Last 20 readings
    }), 200


# ─────────────────────────────────────────────
# ROUTE 3: Get global model status
# GET /global-model
# ─────────────────────────────────────────────
@app.route('/global-model', methods=['GET'])
def get_global_model():
    if global_model is None:
        return jsonify({
            "status": "not ready",
            "message": f"Need at least 2 zones. Currently have: {list(zone_weights.keys())}"
        }), 200

    return jsonify({
        "status": "ready",
        "zones_used": list(zone_weights.keys()),
        "total_readings": len(all_readings),
        "message": "Global model is trained and ready"
    }), 200


# ─────────────────────────────────────────────
# ROUTE 4: Health check
# GET /
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "project": "Critical Water Infrastructure Anomaly Detection",
        "status": "Cloud server running",
        "endpoints": {
            "POST /send-data": "Send sensor data from laptop",
            "GET /data":       "View all stored readings",
            "GET /global-model": "Check global model status"
        }
    }), 200


# ─────────────────────────────────────────────
# FEDERATED AVERAGING - Build Global Model
# ─────────────────────────────────────────────
def build_global_model():
    global global_model, global_scaler

    print(f"\n[CLOUD] Building Global Model from {len(zone_weights)} zones...")

    # Average the model weights from all zones
    all_weights = list(zone_weights.values())
    averaged_weights = np.mean(all_weights, axis=0)

    # Build synthetic training data from averaged weights
    # (simulates federated averaging result)
    np.random.seed(42)
    synthetic_data = np.random.randn(300, 3) * averaged_weights[:3] if len(averaged_weights) >= 3 else np.random.randn(300, 3)

    global_scaler.fit(synthetic_data)
    scaled = global_scaler.transform(synthetic_data)

    global_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    global_model.fit(scaled)

    print(f"[CLOUD] ✅ Global Model ready! Trained from zones: {list(zone_weights.keys())}\n")


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
