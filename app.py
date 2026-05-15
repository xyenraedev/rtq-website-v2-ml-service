"""
Flask ML Service — endpoint REST untuk Decision Tree BBK/TBBK

Endpoint:
  GET  /health
  POST /klasifikasi
  POST /klasifikasi/batch
  POST /latih
  GET  /model/info
  GET  /model/feature-importance
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

from model import DecisionTreeModel

app = Flask(__name__)
CORS(app)

# Singleton model — dimuat sekali saat server naik
_model = DecisionTreeModel()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_trained": _model.is_trained,
            "model_versi": _model.versi,
            "total_data_latih": _model.total_data_latih,
        }
    )


# ---------------------------------------------------------------------------
# Klasifikasi tunggal
# ---------------------------------------------------------------------------


@app.post("/klasifikasi")
def klasifikasi():
    body: dict = request.get_json(force=True) or {}

    try:
        hasil = _model.klasifikasi(body)
        return jsonify(hasil)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Internal error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Klasifikasi batch
# ---------------------------------------------------------------------------


@app.post("/klasifikasi/batch")
def klasifikasi_batch():
    body: dict = request.get_json(force=True) or {}
    santri_list: list[dict] = body.get("santri_list", [])

    if not isinstance(santri_list, list):
        return jsonify({"error": "santri_list harus berupa array"}), 400

    hasil: list[dict] = []
    berhasil = 0
    gagal = 0

    for santri in santri_list:
        santri_id = santri.get("id", "")
        try:
            res = _model.klasifikasi(santri)
            hasil.append({"id": santri_id, "success": True, **res})
            berhasil += 1
        except Exception as exc:  # noqa: BLE001
            hasil.append({"id": santri_id, "success": False, "error": str(exc)})
            gagal += 1

    return jsonify({"hasil": hasil, "berhasil": berhasil, "gagal": gagal})


# ---------------------------------------------------------------------------
# Latih ulang model
# ---------------------------------------------------------------------------


@app.post("/latih")
def latih():
    body: dict = request.get_json(force=True) or {}
    aturan: dict = body.get("aturan", {})
    data_latih: list[dict] | None = body.get("data_latih")

    if not aturan:
        return jsonify({"error": "Field 'aturan' wajib diisi"}), 400

    required_keys = [
        "batas_durasi_jilid_0_4",
        "batas_durasi_jilid_5_6",
        "batas_pengulangan_taskih",
    ]
    for key in required_keys:
        if key not in aturan:
            return jsonify({"error": f"Field aturan.{key} wajib diisi"}), 400

    try:
        hasil = _model.latih(aturan, data_latih=data_latih)
        return jsonify(hasil)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Gagal melatih model: {exc}"}), 500


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


@app.get("/model/info")
def model_info():
    return jsonify(_model.get_info())


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


@app.get("/model/feature-importance")
def feature_importance():
    try:
        return jsonify(_model.get_feature_importance())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
