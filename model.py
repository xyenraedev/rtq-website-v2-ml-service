"""
Decision Tree Model untuk Klasifikasi Santri

Klasifikasi:
  BBK  = Butuh Bimbingan Khusus     → TIDAK memenuhi aturan capaian
  TBBK = Tidak Butuh Bimbingan Khusus → memenuhi aturan capaian

Penamaan versi model: model-A-YYYYMMDD (mudah dibaca & profesional)
  Contoh: model-A-20240115, model-B-20240115, dst.
"""

from __future__ import annotations

import os
import string
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

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
_HURUF = list(string.ascii_uppercase)  # A–Z untuk penomoran versi


# ---------------------------------------------------------------------------
# Helper: penamaan versi profesional
# ---------------------------------------------------------------------------

def _buat_versi(tanggal: str | None = None) -> str:
    """
    Buat nama versi model yang mudah dibaca.

    Format: model-<HURUF>-<YYYYMMDD>
    Huruf bertambah setiap kali model dilatih ulang pada hari yang sama.
    Contoh: model-A-20240115, model-B-20240115, model-A-20240116
    """
    hari_ini = tanggal or datetime.now().strftime("%Y%m%d")

    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            versi_lama: str = data.get("versi", "")
            # Cek apakah versi lama untuk hari yang sama
            if versi_lama.endswith(f"-{hari_ini}"):
                # Naikkan huruf
                bagian = versi_lama.split("-")  # ["model", "A", "20240115"]
                huruf_lama = bagian[1] if len(bagian) >= 3 else "A"
                idx = _HURUF.index(huruf_lama) if huruf_lama in _HURUF else -1
                huruf_baru = _HURUF[idx + 1] if idx + 1 < len(_HURUF) else "A"
                return f"model-{huruf_baru}-{hari_ini}"
        except Exception:
            pass

    return f"model-A-{hari_ini}"


# ---------------------------------------------------------------------------
# DecisionTreeModel
# ---------------------------------------------------------------------------


class DecisionTreeModel:
    """Decision Tree classifier untuk klasifikasi BBK / TBBK santri."""

    def __init__(self) -> None:
        self.clf: DecisionTreeClassifier | None = None
        self.is_trained: bool = False
        self.versi: str = "belum-dilatih"
        self.total_data_latih: int = 0
        self.aturan_aktif: dict = {}
        self.feature_importances_: list[tuple[str, float]] = []

        if os.path.exists(MODEL_PATH):
            self._load()

    # -----------------------------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------------------------

    def _extract_features(self, santri: dict) -> np.ndarray:
        """Ubah data santri menjadi vektor fitur numpy."""
        durasi_list: list[float] = []

        for i in range(7):
            val = santri.get(f"durasi_jilid_{i}")
            durasi_list.append(float(val) if val is not None else 0.0)

        jilid = float(santri.get("jilid_saat_ini", 0))
        taskih = float(santri.get("total_pengulangan_taskih", 0))

        durasi_diambil = [d for d in durasi_list if d > 0]
        rata_rata = float(np.mean(durasi_diambil)) if durasi_diambil else 0.0
        total_durasi = sum(durasi_list)
        jumlah_jilid = float(len(durasi_diambil))

        fitur = [jilid, taskih, *durasi_list, rata_rata, total_durasi, jumlah_jilid]
        return np.array(fitur, dtype=float)

    # -----------------------------------------------------------------------
    # Generate Synthetic Training Data
    # -----------------------------------------------------------------------

    def _generate_data_latih(
        self,
        aturan: dict,
        n_samples: int = 800,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data berdasarkan aturan capaian.

        Logika label:
          TBBK → semua jilid yang diambil memenuhi batas durasi DAN
                  pengulangan taskih di bawah batas
          BBK  → salah satu jilid melebihi batas ATAU taskih melebihi batas
        """
        batas_04 = float(aturan.get("batas_durasi_jilid_0_4", 3))
        batas_56 = float(aturan.get("batas_durasi_jilid_5_6", 4))
        batas_taskih = float(aturan.get("batas_pengulangan_taskih", 2))

        X: list[list[float]] = []
        y: list[str] = []

        rng = np.random.default_rng(42)

        for _ in range(n_samples):
            jilid = int(rng.integers(0, 8))      # 0–7 (7 = Al-Quran)
            taskih = int(rng.integers(0, 6))

            durasi_list: list[float] = []

            for i in range(7):
                if i <= jilid:
                    batas_i = batas_04 if i <= 4 else batas_56
                    # 60% cepat (memenuhi batas), 40% lambat (melampaui batas)
                    if rng.random() < 0.6:
                        durasi = rng.uniform(0.5, batas_i)
                    else:
                        durasi = rng.uniform(batas_i, batas_i * 3)
                    durasi_list.append(round(float(durasi), 1))
                else:
                    durasi_list.append(0.0)

            durasi_diambil = [d for d in durasi_list if d > 0]
            rata_rata = float(np.mean(durasi_diambil)) if durasi_diambil else 0.0
            total_durasi = sum(durasi_list)
            jumlah_jilid = float(len(durasi_diambil))

            # ── LOGIKA LABEL (BENAR) ──────────────────────────────────────
            # BBK  = ada jilid yang melebihi batas ATAU taskih melebihi batas
            # TBBK = semua jilid memenuhi batas DAN taskih memenuhi batas
            ada_yang_melebihi = False
            for i, d in enumerate(durasi_list):
                if d <= 0:
                    continue
                batas_i = batas_04 if i <= 4 else batas_56
                if d > batas_i:
                    ada_yang_melebihi = True
                    break

            taskih_melebihi = taskih >= batas_taskih

            is_bbk = ada_yang_melebihi or taskih_melebihi
            label = "BBK" if is_bbk else "TBBK"

            # Noise realistis 5 %
            if rng.random() < 0.05:
                label = "TBBK" if label == "BBK" else "BBK"

            fitur = [float(jilid), float(taskih), *durasi_list, rata_rata, total_durasi, jumlah_jilid]
            X.append(fitur)
            y.append(label)

        return np.array(X), np.array(y)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def latih(
        self,
        aturan: dict,
        data_latih: list[dict] | None = None,
    ) -> dict:
        """
        Latih ulang Decision Tree.

        Parameters
        ----------
        aturan      : dict berisi batas_durasi_jilid_0_4, batas_durasi_jilid_5_6,
                      batas_pengulangan_taskih
        data_latih  : list data real (opsional, min 10 baris)

        Returns
        -------
        dict berisi versi, akurasi, precision, recall, f1, total_data_latih,
             total_data_test, berhasil
        """
        self.aturan_aktif = aturan

        if data_latih and len(data_latih) >= 10:
            print(f"  Melatih dengan {len(data_latih)} data real...")
            X_list: list[np.ndarray] = []
            y_list: list[str] = []

            for row in data_latih:
                try:
                    fitur = self._extract_features(row)
                    label = row.get("label", row.get("status", ""))
                    if label in ("BBK", "TBBK"):
                        X_list.append(fitur)
                        y_list.append(label)
                except Exception:  # noqa: BLE001
                    continue

            X = np.array(X_list)
            y = np.array(y_list)
        else:
            print("  Menggunakan synthetic data (belum ada data real)...")
            X, y = self._generate_data_latih(aturan, n_samples=800)

        # Split train / test
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Inisialisasi & latih Decision Tree
        self.clf = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            criterion="gini",
            class_weight="balanced",
            random_state=42,
        )
        self.clf.fit(X_train, y_train)

        # Evaluasi
        y_pred = self.clf.predict(X_test)

        akurasi = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        recall = float(recall_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label="BBK", zero_division=0))

        # Metadata
        self.is_trained = True
        self.total_data_latih = len(X)
        self.versi = _buat_versi()
        self.feature_importances_ = list(
            zip(FEATURE_NAMES, self.clf.feature_importances_.tolist())
        )

        self._save()

        result = {
            "versi": self.versi,
            "akurasi": round(akurasi, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "berhasil": int(len(X)),
            "total_data_latih": int(len(X)),
            "total_data_test": int(len(X_test)),
        }

        print(
            f"  ✅ Selesai: akurasi={akurasi:.3f}, "
            f"f1={f1:.3f}, versi={self.versi}"
        )

        return result

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def klasifikasi(self, santri: dict) -> dict:
        """Klasifikasi satu santri → BBK atau TBBK."""
        if not self.is_trained or self.clf is None:
            raise ValueError("Model belum dilatih.")

        fitur = self._extract_features(santri)
        fitur_2d = fitur.reshape(1, -1)

        label: str = str(self.clf.predict(fitur_2d)[0])
        proba = self.clf.predict_proba(fitur_2d)[0]
        classes = list(self.clf.classes_)
        idx = classes.index(label)
        probabilitas = float(proba[idx])

        alasan = self._buat_alasan(santri, label, fitur)

        fitur_snapshot = {
            FEATURE_NAMES[i]: round(float(fitur[i]), 2)
            for i in range(len(FEATURE_NAMES))
        }

        return {
            "status": label,
            "probabilitas": round(probabilitas, 4),
            "alasan": alasan,
            "fitur_snapshot": fitur_snapshot,
            "model_versi": self.versi,
        }

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------

    def _buat_alasan(
        self,
        santri: dict,
        label: str,
        _fitur: np.ndarray,
    ) -> str:
        """Buat kalimat penjelasan hasil klasifikasi."""
        jilid = int(santri.get("jilid_saat_ini", 0))
        taskih = int(santri.get("total_pengulangan_taskih", 0))

        durasi_diambil = [
            float(santri.get(f"durasi_jilid_{i}", 0) or 0)
            for i in range(7)
            if (santri.get(f"durasi_jilid_{i}") or 0) > 0
        ]

        rata_rata = round(float(np.mean(durasi_diambil)), 1) if durasi_diambil else 0
        jilid_label = "Al-Quran" if jilid == 7 else f"Jilid {jilid}"

        # Detail per jilid
        batas_04 = float(self.aturan_aktif.get("batas_durasi_jilid_0_4", 3))
        batas_56 = float(self.aturan_aktif.get("batas_durasi_jilid_5_6", 4))
        batas_taskih = float(self.aturan_aktif.get("batas_pengulangan_taskih", 2))

        detail_parts: list[str] = []
        for i in range(7):
            val = santri.get(f"durasi_jilid_{i}")
            if val is None or float(val) <= 0:
                continue
            batas_i = batas_04 if i <= 4 else batas_56
            val_f = float(val)
            tanda = "❌" if val_f > batas_i else "✓"
            detail_parts.append(f"Jilid {i}: {val_f} bln (batas {batas_i} bln) {tanda}")

        taskih_tanda = "❌" if taskih >= batas_taskih else "✓"
        detail_parts.append(f"Taskih: {taskih}x (batas {int(batas_taskih)}x) {taskih_tanda}")

        detail_str = " | ".join(detail_parts)

        if label == "BBK":
            ringkasan = (
                f"Santri pada {jilid_label} dengan rata-rata durasi "
                f"{rata_rata} bln/jilid dan {taskih}x pengulangan taskih "
                f"terindikasi MEMBUTUHKAN bimbingan khusus (BBK)."
            )
        else:
            ringkasan = (
                f"Santri pada {jilid_label} dengan rata-rata durasi "
                f"{rata_rata} bln/jilid dan {taskih}x pengulangan taskih "
                f"dinilai TIDAK membutuhkan bimbingan khusus (TBBK)."
            )

        return f"{ringkasan}\n\nDetail: {detail_str}"

    # -----------------------------------------------------------------------
    # Info & Feature Importance
    # -----------------------------------------------------------------------

    def get_info(self) -> dict:
        """Kembalikan metadata model."""
        return {
            "is_trained": self.is_trained,
            "versi": self.versi,
            "total_data_latih": self.total_data_latih,
            "aturan_aktif": self.aturan_aktif,
            "feature_names": FEATURE_NAMES,
            "algorithm": "DecisionTreeClassifier (scikit-learn)",
            "params": (
                {
                    "max_depth": 8,
                    "min_samples_split": 5,
                    "min_samples_leaf": 3,
                    "criterion": "gini",
                }
                if self.clf
                else {}
            ),
        }

    def get_feature_importance(self) -> dict:
        """Kembalikan feature importance terurut dari yang tertinggi."""
        if not self.is_trained:
            raise ValueError("Model belum dilatih.")

        return {
            "features": [
                {"nama": name, "importance": round(float(imp), 4)}
                for name, imp in sorted(
                    self.feature_importances_,
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
        }

    # -----------------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------------

    def _save(self) -> None:
        joblib.dump(
            {
                "clf": self.clf,
                "versi": self.versi,
                "total_data_latih": self.total_data_latih,
                "aturan_aktif": self.aturan_aktif,
                "feature_importances_": self.feature_importances_,
            },
            MODEL_PATH,
        )
        print(f"  💾 Model disimpan ke {MODEL_PATH} ({self.versi})")

    def _load(self) -> None:
        try:
            data: dict = joblib.load(MODEL_PATH)
            self.clf = data["clf"]
            self.versi = data["versi"]
            self.total_data_latih = data["total_data_latih"]
            self.aturan_aktif = data.get("aturan_aktif", {})
            self.feature_importances_ = data.get("feature_importances_", [])
            self.is_trained = True
            print(f"  📂 Model dimuat dari disk: {self.versi}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  Gagal load model: {exc}")
