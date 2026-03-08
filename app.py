"""
ML Service - Decision Tree Classifier untuk Santri
Flask REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

from model import DecisionTreeModel

app = Flask(__name__)
CORS(app)  # Izinkan request dari Next.js

# Inisialisasi model (singleton)
model = DecisionTreeModel()


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": model.is_trained,
        "model_versi": model.versi,
        "total_data_latih": model.total_data_latih,
    })


# ─── Klasifikasi Satu Santri ──────────────────────────────────────────────────

@app.route('/klasifikasi', methods=['POST'])
def klasifikasi():
    """
    Input JSON:
    {
        "jilid_saat_ini": 3,
        "total_pengulangan_taskih": 1,
        "durasi_jilid_0": 2,
        "durasi_jilid_1": 3,
        "durasi_jilid_2": null,
        ...
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Body JSON tidak boleh kosong"}), 400

        hasil = model.klasifikasi(data)
        return jsonify(hasil)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ─── Klasifikasi Batch (banyak santri sekaligus) ──────────────────────────────

@app.route('/klasifikasi/batch', methods=['POST'])
def klasifikasi_batch():
    """
    Input JSON:
    {
        "santri_list": [
            {"id": "uuid1", "jilid_saat_ini": 3, ...},
            {"id": "uuid2", "jilid_saat_ini": 5, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or "santri_list" not in data:
            return jsonify({"error": "Field 'santri_list' wajib ada"}), 400

        santri_list = data["santri_list"]
        if not isinstance(santri_list, list):
            return jsonify({"error": "'santri_list' harus berupa array"}), 400

        hasil_list = []
        berhasil = 0
        gagal = 0

        for santri in santri_list:
            santri_id = santri.get("id", "unknown")
            try:
                hasil = model.klasifikasi(santri)
                hasil_list.append({
                    "id": santri_id,
                    "success": True,
                    **hasil
                })
                berhasil += 1
            except Exception as e:
                hasil_list.append({
                    "id": santri_id,
                    "success": False,
                    "error": str(e)
                })
                gagal += 1

        return jsonify({
            "hasil": hasil_list,
            "berhasil": berhasil,
            "gagal": gagal,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ─── Latih Ulang Model ────────────────────────────────────────────────────────

@app.route('/latih', methods=['POST'])
def latih():
    """
    Input JSON:
    {
        "aturan": {
            "batas_durasi_jilid_0_4": 3,
            "batas_durasi_jilid_5_6": 4,
            "batas_pengulangan_taskih": 2
        },
        "data_latih": [   <-- opsional, kalau tidak ada akan pakai synthetic data
            {
                "jilid_saat_ini": 3,
                "total_pengulangan_taskih": 1,
                "durasi_jilid_0": 2,
                "label": "BBK"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Body JSON tidak boleh kosong"}), 400

        aturan = data.get("aturan", {})
        data_latih = data.get("data_latih", None)

        hasil_evaluasi = model.latih(aturan=aturan, data_latih=data_latih)
        return jsonify(hasil_evaluasi)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ─── Info Model ───────────────────────────────────────────────────────────────

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify(model.get_info())


# ─── Feature Importance ───────────────────────────────────────────────────────

@app.route('/model/feature-importance', methods=['GET'])
def feature_importance():
    try:
        return jsonify(model.get_feature_importance())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Auto-latih dengan aturan default saat startup
    print("🚀 Melatih model Decision Tree dengan data awal...")
    model.latih(aturan={
        "batas_durasi_jilid_0_4": 3,
        "batas_durasi_jilid_5_6": 4,
        "batas_pengulangan_taskih": 2,
    })
    print("✅ Model siap!")
    app.run(host='0.0.0.0', port=5000, debug=True)  
