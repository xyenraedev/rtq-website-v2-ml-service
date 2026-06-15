"""
Decision Tree Model untuk Klasifikasi Santri
============================================

Klasifikasi:
  BBK  = Butuh Bimbingan Khusus      → TIDAK memenuhi aturan capaian
  TBBK = Tidak Butuh Bimbingan Khusus → memenuhi aturan capaian

Penamaan versi model: model-A-YYYYMMDD
  Contoh: model-A-20240115, model-B-20240115, dst.

ARSITEKTUR DATA LATIH
─────────────────────
Sumber data latih utama: tabel training_master (di-generate oleh trigger SQL
`generate_training_master` saat aturan baru disimpan).

Fungsi `_generate_hardcoded_data` di sini adalah MIRROR dari logika SQL tersebut
— digunakan sebagai fallback jika training_master kosong atau belum tersedia.
Keduanya menggunakan 5 skenario yang sama agar perilaku konsisten:

  1. TBBK Jelas      — durasi & taskih jauh di bawah batas
  2. BBK Durasi      — durasi jauh di atas batas, taskih rendah
  3. BBK Taskih      — durasi bagus, taskih jauh di atas batas
  4. BBK Keduanya    — durasi & taskih melebihi batas
  5. Zona Abu-abu    — nilai di sekitar ±10% threshold (edge case nyata)

Semua nilai dinyatakan RELATIF terhadap aturan aktif sehingga model
otomatis menyesuaikan jika admin mengubah batas di website.
"""

from __future__ import annotations

import os
import string
import warnings
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "jilid_saat_ini",
    "total_pengulangan_taskih",
    "durasi_jilid_0",
    "durasi_jilid_1",
    "durasi_jilid_2",
    "durasi_jilid_3",
    "durasi_jilid_4",
    "durasi_jilid_5",
    "durasi_jilid_6",
    "rata_rata_durasi",
    "total_durasi",
    "jumlah_jilid_diambil",
]

MODEL_PATH = "model.joblib"
_HURUF     = list(string.ascii_uppercase)

# Lebar zona abu-abu: ±10% di sekitar threshold
# Harus sama dengan v_overlap_ratio di SQL trigger
_OVERLAP_RATIO = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Helper: penamaan versi profesional
# ─────────────────────────────────────────────────────────────────────────────

def _buat_versi(tanggal: str | None = None) -> str:
    """Format: model-<HURUF>-<YYYYMMDD>. Huruf naik jika dilatih ulang hari sama."""
    hari_ini = tanggal or datetime.now().strftime("%Y%m%d")

    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            versi_lama: str = data.get("versi", "")
            if versi_lama.endswith(f"-{hari_ini}"):
                bagian    = versi_lama.split("-")
                huruf_lama = bagian[1] if len(bagian) >= 3 else "A"
                idx        = _HURUF.index(huruf_lama) if huruf_lama in _HURUF else -1
                huruf_baru = _HURUF[idx + 1] if idx + 1 < len(_HURUF) else "A"
                return f"model-{huruf_baru}-{hari_ini}"
        except Exception:
            pass

    return f"model-A-{hari_ini}"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: bangun satu baris fitur lengkap dari data mentah per jilid
# ─────────────────────────────────────────────────────────────────────────────

def _build_row(
    jilid: int,
    durasi_jilid_aktif: float,
    taskih: int,
) -> list[float]:
    """
    Bangun vektor fitur 12 dimensi untuk satu santri.
    Hanya jilid aktif yang memiliki durasi; jilid sebelumnya diasumsikan
    memiliki durasi rata-rata proporsional (sederhana tapi cukup untuk training).
    Jilid di atas jilid aktif = 0.
    """
    durasi_list = [0.0] * 7
    for i in range(min(jilid + 1, 7)):
        # Jilid sebelumnya: diasumsikan durasi sama dengan jilid aktif
        # (penyederhanaan yang wajar untuk data training)
        durasi_list[i] = durasi_jilid_aktif

    durasi_diambil = [d for d in durasi_list if d > 0]
    rata_rata      = float(np.mean(durasi_diambil)) if durasi_diambil else 0.0
    total_durasi   = sum(durasi_list)
    jumlah_jilid   = float(len(durasi_diambil))

    return [
        float(jilid), float(taskih),
        *durasi_list,
        rata_rata, total_durasi, jumlah_jilid,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DecisionTreeModel
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTreeModel:
    """Decision Tree classifier untuk klasifikasi BBK / TBBK santri."""

    def __init__(self) -> None:
        self.clf: DecisionTreeClassifier | None = None
        self.is_trained:          bool  = False
        self.versi:               str   = "belum-dilatih"
        self.total_data_latih:    int   = 0
        self.aturan_aktif:        dict  = {}
        self.feature_importances_: list[tuple[str, float]] = []
        self.cv_scores_:          list[float] = []

        if os.path.exists(MODEL_PATH):
            self._load()

    # ─────────────────────────────────────────────────────────────────────
    # Feature Engineering
    # ─────────────────────────────────────────────────────────────────────

    def _extract_features(self, santri: dict) -> np.ndarray:
        durasi_list: list[float] = []
        for i in range(7):
            val = santri.get(f"durasi_jilid_{i}")
            durasi_list.append(float(val) if val is not None else 0.0)

        jilid  = float(santri.get("jilid_saat_ini", 0))
        taskih = float(santri.get("total_pengulangan_taskih", 0))

        durasi_diambil = [d for d in durasi_list if d > 0]
        rata_rata      = float(np.mean(durasi_diambil)) if durasi_diambil else 0.0
        total_durasi   = sum(durasi_list)
        jumlah_jilid   = float(len(durasi_diambil))

        return np.array(
            [jilid, taskih, *durasi_list, rata_rata, total_durasi, jumlah_jilid],
            dtype=float,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Generate Hardcoded Data  (MIRROR dari logika SQL trigger)
    # ─────────────────────────────────────────────────────────────────────

    def _generate_hardcoded_data(
        self,
        aturan: dict,
        n_per_skenario: int = 12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mirror dari fungsi SQL generate_training_master.
        5 skenario × 8 jilid × n_per_skenario = ~480 baris.

        Skenario:
          1. TBBK Jelas      — durasi 0.5 → batas_low,  taskih rendah
          2. BBK Durasi      — durasi batas_high → batas×3, taskih 0
          3. BBK Taskih      — durasi rendah, taskih batas+1 → batas×3
          4. BBK Keduanya    — durasi & taskih melebihi batas
          5. Zona Abu-abu    — nilai di batas_low → batas_high (edge case)
        """
        b04   = float(aturan.get("batas_durasi_jilid_0_4", 3))
        b56   = float(aturan.get("batas_durasi_jilid_5_6", 4))
        b_tsk = float(aturan.get("batas_pengulangan_taskih", 2))

        X: list[list[float]] = []
        y: list[str]         = []

        def linspace_n(start: float, end: float, n: int) -> list[float]:
            if n == 1:
                return [start]
            return [start + (end - start) * i / (n - 1) for i in range(n)]

        for jilid in range(8):
            batas = b04 if jilid <= 4 else b56
            b_low  = round(batas * (1 - _OVERLAP_RATIO), 2)
            b_high = round(batas * (1 + _OVERLAP_RATIO), 2)

            # ── JILID 7: Al-Quran selalu TBBK ────────────────────────────
            if jilid == 7:
                for d in linspace_n(0.5, batas * 2.5, n_per_skenario):
                    X.append(_build_row(jilid, round(d, 1), 0))
                    y.append("TBBK")
                continue

            # ── SKENARIO 1: TBBK Jelas ────────────────────────────────────
            for i, d in enumerate(linspace_n(0.5, b_low, n_per_skenario)):
                t = int((b_tsk - 1) * i / (n_per_skenario - 1)) if n_per_skenario > 1 else 0
                t = max(0, t)
                X.append(_build_row(jilid, round(d, 1), t))
                y.append("TBBK")

            # ── SKENARIO 1b: TBBK Variasi Taskih Nol ─────────────────────
            # Santri cepat/hafidz, taskih 0, durasi bervariasi rendah
            for d in linspace_n(0.5, b_low * 0.8, n_per_skenario):
                X.append(_build_row(jilid, round(d, 1), 0))
                y.append("TBBK")

            # ── SKENARIO 1c: TBBK Medium ──────────────────────────────────
            # Santri biasa-baik, durasi 50-90% batas, taskih bervariasi
            for i, d in enumerate(linspace_n(batas * 0.5, b_low, n_per_skenario)):
                t = int((b_tsk - 1) * i / (n_per_skenario - 1)) if n_per_skenario > 1 else 0
                t = max(0, t)
                X.append(_build_row(jilid, round(d, 1), t))
                y.append("TBBK")

            # ── SKENARIO 2: BBK karena Durasi ────────────────────────────
            for d in linspace_n(b_high, batas * 3, n_per_skenario):
                X.append(_build_row(jilid, round(d, 1), 0))
                y.append("BBK")

            # ── SKENARIO 3: BBK karena Taskih ────────────────────────────
            for i, d in enumerate(linspace_n(0.5, b_low, n_per_skenario)):
                t = int(b_tsk + 1 + b_tsk * 2 * i / (n_per_skenario - 1)) if n_per_skenario > 1 else int(b_tsk + 1)
                X.append(_build_row(jilid, round(d, 1), t))
                y.append("BBK")

            # ── SKENARIO 4: BBK Durasi + Taskih ──────────────────────────
            for i, d in enumerate(linspace_n(b_high, batas * 2, n_per_skenario)):
                t = int(b_tsk + 1 + b_tsk * 2 * i / (n_per_skenario - 1)) if n_per_skenario > 1 else int(b_tsk + 1)
                X.append(_build_row(jilid, round(d, 1), t))
                y.append("BBK")

            # ── SKENARIO 5: Zona Abu-abu (edge case) ─────────────────────
            for i, d in enumerate(linspace_n(b_low, b_high, n_per_skenario)):
                t = int((b_tsk + 1) * i / (n_per_skenario - 1)) if n_per_skenario > 1 else 0
                d_r = round(d, 2)
                # Label deterministik sesuai aturan (sama seperti SQL)
                label = "BBK" if (d_r > batas or t >= b_tsk) else "TBBK"
                X.append(_build_row(jilid, d_r, t))
                y.append(label)

        X_arr = np.array(X)
        y_arr = np.array(y)

        n_bbk  = int(np.sum(y_arr == "BBK"))
        n_tbbk = int(np.sum(y_arr == "TBBK"))
        print(
            f"  📊 Hardcoded data: {len(y_arr)} baris | "
            f"BBK={n_bbk} ({n_bbk/len(y_arr)*100:.1f}%) | "
            f"TBBK={n_tbbk} ({n_tbbk/len(y_arr)*100:.1f}%)"
        )

        return X_arr, y_arr

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def latih(
        self,
        aturan: dict,
        data_latih: list[dict] | None = None,
    ) -> dict:
        """
        Latih ulang Decision Tree.

        Prioritas sumber data:
          1. data_latih dari training_master (dikirim oleh service TS)
          2. _generate_hardcoded_data sebagai fallback

        Returns dict: versi, akurasi, precision, recall, f1,
                      cv_mean, cv_std, total_data_latih, peringatan
        """
        self.aturan_aktif      = aturan
        menggunakan_data_real  = False

        if data_latih and len(data_latih) >= 10:
            print(f"  Melatih dengan {len(data_latih)} baris dari training_master...")
            menggunakan_data_real = True
            X_list, y_list = [], []
            for row in data_latih:
                try:
                    fitur = self._extract_features(row)
                    label = row.get("label", row.get("status", ""))
                    if label in ("BBK", "TBBK"):
                        X_list.append(fitur)
                        y_list.append(label)
                except Exception:
                    continue
            X = np.array(X_list)
            y = np.array(y_list)
        else:
            print("  training_master kosong — menggunakan hardcoded fallback...")
            X, y = self._generate_hardcoded_data(aturan)

        # ── Split ──────────────────────────────────────────────────────
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # ── Model ──────────────────────────────────────────────────────
        self.clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=15,
            min_samples_leaf=10,
            criterion="gini",
            class_weight="balanced",
            random_state=42,
        )
        self.clf.fit(X_train, y_train)

        # ── Evaluasi hold-out ──────────────────────────────────────────
        y_pred    = self.clf.predict(X_test)
        akurasi   = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        recall    = float(recall_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        f1        = float(f1_score(y_test, y_pred, pos_label="BBK", zero_division=0))

        # ── Cross-validation 5-fold ────────────────────────────────────
        cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.clf, X, y, cv=cv, scoring="accuracy")
        cv_mean   = float(np.mean(cv_scores))
        cv_std    = float(np.std(cv_scores))
        self.cv_scores_ = cv_scores.tolist()

        print(
            f"  CV 5-fold: {cv_mean:.3f} ± {cv_std:.3f}  "
            f"(min={cv_scores.min():.3f}, max={cv_scores.max():.3f})"
        )

        # ── Peringatan ─────────────────────────────────────────────────
        peringatan: list[str] = []

        if not menggunakan_data_real:
            peringatan.append(
                "Model dilatih dengan hardcoded fallback (training_master kosong). "
                "Latih ulang setelah training_master terisi dari trigger SQL."
            )
        if akurasi > 0.97 and not menggunakan_data_real:
            peringatan.append(
                f"Akurasi {akurasi:.1%} wajar untuk data terstruktur. "
                "Pantau performa di data santri asli secara berkala."
            )
        if cv_std > 0.05:
            peringatan.append(
                f"Variansi CV tinggi (std={cv_std:.3f}). "
                "Pertimbangkan menambah n_per_skenario atau memperkecil max_depth."
            )

        for p in peringatan:
            warnings.warn(p, UserWarning, stacklevel=2)

        # ── Metadata ───────────────────────────────────────────────────
        self.is_trained           = True
        self.total_data_latih     = len(X)
        self.versi                = _buat_versi()
        self.feature_importances_ = list(
            zip(FEATURE_NAMES, self.clf.feature_importances_.tolist())
        )
        self._save()

        result = {
            "versi":                 self.versi,
            "akurasi":               round(akurasi, 4),
            "precision":             round(precision, 4),
            "recall":                round(recall, 4),
            "f1":                    round(f1, 4),
            "cv_mean":               round(cv_mean, 4),
            "cv_std":                round(cv_std, 4),
            "berhasil":              int(len(X)),
            "total_data_latih":      int(len(X)),
            "total_data_test":       int(len(X_test)),
            "menggunakan_data_real": menggunakan_data_real,
            "peringatan":            peringatan,
        }

        print(
            f"  ✅ Selesai: akurasi={akurasi:.3f}, f1={f1:.3f}, "
            f"cv={cv_mean:.3f}±{cv_std:.3f}, versi={self.versi}"
        )
        for p in peringatan:
            print(f"  ⚠️  {p}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def klasifikasi(self, santri: dict) -> dict:
        if not self.is_trained or self.clf is None:
            raise ValueError("Model belum dilatih.")

        fitur       = self._extract_features(santri)
        fitur_2d    = fitur.reshape(1, -1)
        label: str  = str(self.clf.predict(fitur_2d)[0])
        proba       = self.clf.predict_proba(fitur_2d)[0]
        classes     = list(self.clf.classes_)
        probabilitas = float(proba[classes.index(label)])
        alasan      = self._buat_alasan(santri, label, fitur)
        fitur_snapshot = {
            FEATURE_NAMES[i]: round(float(fitur[i]), 2)
            for i in range(len(FEATURE_NAMES))
        }

        return {
            "status":         label,
            "probabilitas":   round(probabilitas, 4),
            "alasan":         alasan,
            "fitur_snapshot": fitur_snapshot,
            "model_versi":    self.versi,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Explanation
    # ─────────────────────────────────────────────────────────────────────

    def _buat_alasan(self, santri: dict, label: str, _fitur: np.ndarray) -> str:
        jilid  = int(santri.get("jilid_saat_ini", 0))
        taskih = int(santri.get("total_pengulangan_taskih", 0))

        durasi_diambil = [
            float(santri.get(f"durasi_jilid_{i}", 0) or 0)
            for i in range(7)
            if (santri.get(f"durasi_jilid_{i}") or 0) > 0
        ]

        rata_rata   = round(float(np.mean(durasi_diambil)), 1) if durasi_diambil else 0
        jilid_label = "Al-Quran" if jilid == 7 else f"Jilid {jilid}"

        b04   = float(self.aturan_aktif.get("batas_durasi_jilid_0_4", 3))
        b56   = float(self.aturan_aktif.get("batas_durasi_jilid_5_6", 4))
        b_tsk = float(self.aturan_aktif.get("batas_pengulangan_taskih", 2))

        detail_parts: list[str] = []
        for i in range(7):
            val = santri.get(f"durasi_jilid_{i}")
            if val is None or float(val) <= 0:
                continue
            batas_i = b04 if i <= 4 else b56
            val_f   = float(val)
            tanda   = "❌" if val_f > batas_i else "✓"
            detail_parts.append(f"Jilid {i}: {val_f} bln (batas {batas_i} bln) {tanda}")

        tanda_tsk = "❌" if taskih >= b_tsk else "✓"
        detail_parts.append(f"Taskih: {taskih}x (batas {int(b_tsk)}x) {tanda_tsk}")

        ringkasan = (
            f"Santri pada {jilid_label} dengan rata-rata durasi "
            f"{rata_rata} bln/jilid dan {taskih}x pengulangan taskih "
            + ("terindikasi MEMBUTUHKAN bimbingan khusus (BBK)."
               if label == "BBK"
               else "dinilai TIDAK membutuhkan bimbingan khusus (TBBK).")
        )

        return f"{ringkasan}\n\nDetail: {' | '.join(detail_parts)}"

    # ─────────────────────────────────────────────────────────────────────
    # Info & Feature Importance
    # ─────────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "is_trained":       self.is_trained,
            "versi":            self.versi,
            "total_data_latih": self.total_data_latih,
            "aturan_aktif":     self.aturan_aktif,
            "feature_names":    FEATURE_NAMES,
            "algorithm":        "DecisionTreeClassifier (scikit-learn)",
            "params":           {
                "max_depth": 5, "min_samples_split": 15,
                "min_samples_leaf": 10, "criterion": "gini",
            } if self.clf else {},
            "cv_scores": self.cv_scores_,
        }

    def get_feature_importance(self) -> dict:
        if not self.is_trained:
            raise ValueError("Model belum dilatih.")
        return {
            "features": [
                {"nama": name, "importance": round(float(imp), 4)}
                for name, imp in sorted(
                    self.feature_importances_, key=lambda x: x[1], reverse=True
                )
            ]
        }

    # ─────────────────────────────────────────────────────────────────────
    # Persist
    # ─────────────────────────────────────────────────────────────────────

    def _save(self) -> None:
        joblib.dump(
            {
                "clf":                  self.clf,
                "versi":                self.versi,
                "total_data_latih":     self.total_data_latih,
                "aturan_aktif":         self.aturan_aktif,
                "feature_importances_": self.feature_importances_,
                "cv_scores_":           self.cv_scores_,
            },
            MODEL_PATH,
        )
        print(f"  💾 Model disimpan ke {MODEL_PATH} ({self.versi})")

    def _load(self) -> None:
        try:
            data: dict = joblib.load(MODEL_PATH)
            self.clf                  = data["clf"]
            self.versi                = data["versi"]
            self.total_data_latih     = data["total_data_latih"]
            self.aturan_aktif         = data.get("aturan_aktif", {})
            self.feature_importances_ = data.get("feature_importances_", [])
            self.cv_scores_           = data.get("cv_scores_", [])
            self.is_trained           = True
            print(f"  📂 Model dimuat dari disk: {self.versi}")
        except Exception as exc:
            print(f"  ⚠️  Gagal load model: {exc}")
