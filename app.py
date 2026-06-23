from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from model import DecisionTreeModel, TREE_IMAGE_PATH

app = Flask(__name__)
CORS(app)

_model = DecisionTreeModel()


@app.get("/health")
def health():
    return jsonify({
        "status":           "ok",
        "model_trained":    _model.is_trained,
        "model_versi":      _model.versi,
        "total_data_latih": _model.total_data_latih,
    })


@app.post("/klasifikasi")
def klasifikasi():
    body: dict = request.get_json(force=True) or {}

    # 'aturan' adalah aturan_capaian yang sedang is_active=true di DB,
    # WAJIB dikirim oleh caller (lihat mlClient.ts mlKlasifikasi).
    # Tanpa ini, endpoint tidak punya cara tahu aturan mana yang aktif
    # SEKARANG — dan akan diam-diam memakai aturan hasil training
    # terakhir, yang bisa berbeda dari aturan aktif di DB.
    aturan: dict | None = body.pop("aturan", None)

    try:
        return jsonify(_model.klasifikasi(body, aturan=aturan))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Internal error: {exc}"}), 500


@app.post("/klasifikasi/batch")
def klasifikasi_batch():
    body: dict        = request.get_json(force=True) or {}
    santri_list: list = body.get("santri_list", [])
    aturan: dict | None = body.get("aturan")  # aturan aktif, sama untuk semua santri dalam batch

    if not isinstance(santri_list, list):
        return jsonify({"error": "santri_list harus berupa array"}), 400

    hasil: list[dict] = []
    berhasil = 0
    gagal    = 0

    for santri in santri_list:
        santri_id = santri.get("id", "")
        try:
            res = _model.klasifikasi(santri, aturan=aturan)
            hasil.append({"id": santri_id, "success": True, **res})
            berhasil += 1
        except Exception as exc:
            hasil.append({"id": santri_id, "success": False, "error": str(exc)})
            gagal += 1

    return jsonify({"hasil": hasil, "berhasil": berhasil, "gagal": gagal})


@app.post("/latih")
def latih():
    body: dict      = request.get_json(force=True) or {}
    aturan: dict    = body.get("aturan", {})
    data_latih: list | None = body.get("data_latih")

    if not aturan:
        return jsonify({"error": "Field 'aturan' wajib diisi"}), 400

    required_keys = ["batas_durasi_jilid_0_4", "batas_durasi_jilid_5_6", "batas_pengulangan_taskih"]
    for key in required_keys:
        if key not in aturan:
            return jsonify({"error": f"Field aturan.{key} wajib diisi"}), 400

    try:
        return jsonify(_model.latih(aturan, data_latih=data_latih))
    except Exception as exc:
        return jsonify({"error": f"Gagal melatih model: {exc}"}), 500


@app.get("/model/info")
def model_info():
    return jsonify(_model.get_info())


@app.get("/model/feature-importance")
def feature_importance():
    try:
        return jsonify(_model.get_feature_importance())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/model/report")
def model_report():
    """
    Mengembalikan laporan lengkap hasil pelatihan model sesuai Bab 4.1.3:
    - Konfigurasi parameter (Tabel 4.4)
    - Grid Search top-10 (Tabel 4.5)
    - Distribusi dataset stratified split (Tabel 4.6)
    - Hasil evaluasi hold-out: akurasi, presisi, recall, F1 (Tabel 4.7)
    - Hasil 5-fold cross validation per fold (Tabel 4.8)
    - Confusion matrix TP/FN/FP/TN (Tabel 4.9)
    - Feature importance terurut (Tabel 4.10)
    - Path gambar visualisasi pohon keputusan (Gambar 4.8)
    """
    try:
        return jsonify(_model.get_report())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/model/tree-image")
def tree_image():
    """
    Mengembalikan file PNG visualisasi pohon keputusan (Gambar 4.8).
    Gambar di-generate otomatis saat model selesai dilatih.
    """
    if not os.path.exists(TREE_IMAGE_PATH):
        return jsonify({"error": "Visualisasi pohon belum tersedia. Latih model terlebih dahulu."}), 404
    return send_file(TREE_IMAGE_PATH, mimetype="image/png")


@app.get("/model/tree-text")
def tree_text():
    """
    Mengembalikan representasi teks ASCII dari pohon keputusan.
    Berguna untuk debugging atau tampilan alternatif di frontend.
    """
    try:
        return jsonify({"tree_text": _model.get_tree_text()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
