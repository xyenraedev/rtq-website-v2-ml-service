"""
Decision Tree Model untuk Klasifikasi Santri (BBK / TBBK)

BBK  = Bisa Baca Quran (layak naik)
TBBK = Tidak Bisa Baca Quran (perlu pengulangan)
"""

import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Nama fitur yang digunakan model
FEATURE_NAMES = [
    "jilid_saat_ini",
    "total_pengulangan_taskih",
    "durasi_jilid_0",
    "durasi_jilid_1",
    "durasi_jilid_2",
    "durasi_jilid_3",
    "durasi_jilid_4",
    "durasi_jilid_5",
    "durasi_jilid_6",
    "rata_rata_durasi",          # fitur turunan
    "total_durasi",              # fitur turunan
    "jumlah_jilid_diambil",      # fitur turunan
]

MODEL_PATH = "model.joblib"


class DecisionTreeModel:
    def __init__(self):
        self.clf = None
        self.is_trained = False
        self.versi = "belum-dilatih"
        self.total_data_latih = 0
        self.aturan_aktif = {}
        self.feature_importances_ = []

        # Coba load model yang sudah tersimpan
        if os.path.exists(MODEL_PATH):
            self._load()

    # ─── Feature Engineering ──────────────────────────────────────────────────

    def _extract_features(self, santri: dict) -> np.ndarray:
        """Ubah data santri menjadi vektor fitur untuk model."""
        durasi_list = []
        for i in range(7):
            val = santri.get(f"durasi_jilid_{i}")
            # Kalau null/None, isi 0 (tidak mengambil jilid itu)
            durasi_list.append(float(val) if val is not None else 0.0)

        jilid = float(santri.get("jilid_saat_ini", 0))
        taskih = float(santri.get("total_pengulangan_taskih", 0))

        # Fitur turunan
        durasi_diambil = [d for d in durasi_list if d > 0]
        rata_rata = np.mean(durasi_diambil) if durasi_diambil else 0.0
        total_durasi = sum(durasi_list)
        jumlah_jilid = len(durasi_diambil)

        fitur = [
            jilid,
            taskih,
            *durasi_list,
            rata_rata,
            total_durasi,
            jumlah_jilid,
        ]

        return np.array(fitur, dtype=float)

    # ─── Generate Synthetic Training Data ────────────────────────────────────

    def _generate_data_latih(self, aturan: dict, n_samples: int = 500) -> tuple:
        """
        Buat data sintetis berdasarkan aturan rule-based.
        Ini digunakan sebagai data latih awal sebelum ada data real dari guru.

        Logika BBK:
        - Jilid 0-4: rata-rata durasi <= batas_durasi_jilid_0_4
        - Jilid 5-6: rata-rata durasi <= batas_durasi_jilid_5_6
        - pengulangan taskih < batas_pengulangan_taskih
        """
        batas_04 = float(aturan.get("batas_durasi_jilid_0_4", 3))
        batas_56 = float(aturan.get("batas_durasi_jilid_5_6", 4))
        batas_taskih = float(aturan.get("batas_pengulangan_taskih", 2))

        X = []
        y = []

        np.random.seed(42)

        for _ in range(n_samples):
            jilid = np.random.randint(0, 8)  # 0-7 (7 = Al-Quran)
            taskih = np.random.randint(0, 6)

            # Generate durasi per jilid sesuai jilid saat ini
            durasi_list = []
            for i in range(7):
                if i <= jilid:
                    # Variasi: mix antara cepat dan lambat
                    if np.random.random() < 0.6:
                        durasi = np.random.uniform(1, batas_04 + 1)  # cepat
                    else:
                        durasi = np.random.uniform(batas_04, batas_04 * 3)  # lambat
                    durasi_list.append(round(durasi, 1))
                else:
                    durasi_list.append(0.0)

            # Tentukan label berdasarkan rule
            durasi_diambil = [d for d in durasi_list if d > 0]
            rata_rata = np.mean(durasi_diambil) if durasi_diambil else 0.0
            total_durasi = sum(durasi_list)
            jumlah_jilid = len(durasi_diambil)

            # Logika label
            if jilid <= 4:
                batas = batas_04
            else:
                batas = batas_56

            is_bbk = (rata_rata <= batas) and (taskih < batas_taskih)
            label = "BBK" if is_bbk else "TBBK"

            # Tambah noise realistis (5%)
            if np.random.random() < 0.05:
                label = "TBBK" if label == "BBK" else "BBK"

            fitur = [
                float(jilid),
                float(taskih),
                *[float(d) for d in durasi_list],
                rata_rata,
                total_durasi,
                float(jumlah_jilid),
            ]

            X.append(fitur)
            y.append(label)

        return np.array(X), np.array(y)

    # ─── Training ─────────────────────────────────────────────────────────────

    def latih(self, aturan: dict, data_latih: list = None) -> dict:
        """
        Latih model Decision Tree.

        Args:
            aturan: Parameter threshold dari aturan_capaian
            data_latih: List dict santri dengan field 'label' (opsional)

        Returns:
            dict dengan metrik evaluasi
        """
        self.aturan_aktif = aturan

        if data_latih and len(data_latih) >= 10:
            # Pakai data real dari database
            print(f"  Melatih dengan {len(data_latih)} data real...")
            X_list = []
            y_list = []
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
            # Pakai synthetic data
            print("  Menggunakan synthetic data (belum ada data real)...")
            X, y = self._generate_data_latih(aturan, n_samples=800)

        # Split train/test
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Latih Decision Tree
        self.clf = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            criterion="gini",
            class_weight="balanced",  # Handle imbalanced data
            random_state=42,
        )
        self.clf.fit(X_train, y_train)

        # Evaluasi
        y_pred = self.clf.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="BBK", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="BBK", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="BBK", zero_division=0)

        # Simpan metadata
        self.is_trained = True
        self.total_data_latih = len(X)
        self.versi = f"dt-v{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.feature_importances_ = list(zip(
            FEATURE_NAMES,
            self.clf.feature_importances_.tolist()
        ))

        # Simpan model ke disk
        self._save()

        result = {
            "versi": self.versi,
            "akurasi": round(float(akurasi), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "berhasil": int(len(X)),
            "total_data_latih": int(len(X)),
            "total_data_test": int(len(X_test)),
        }

        print(f"  ✅ Selesai: akurasi={akurasi:.3f}, f1={f1:.3f}, versi={self.versi}")
        return result

    # ─── Inference ────────────────────────────────────────────────────────────

    def klasifikasi(self, santri: dict) -> dict:
        """
        Klasifikasi satu santri.

        Returns:
            {
                "status": "BBK" | "TBBK",
                "probabilitas": 0.87,
                "alasan": "...",
                "fitur_snapshot": {...},
                "model_versi": "dt-v..."
            }
        """
        if not self.is_trained or self.clf is None:
            raise ValueError("Model belum dilatih. Panggil /latih terlebih dahulu.")

        fitur = self._extract_features(santri)
        fitur_2d = fitur.reshape(1, -1)

        # Prediksi
        label = self.clf.predict(fitur_2d)[0]
        proba = self.clf.predict_proba(fitur_2d)[0]
        classes = self.clf.classes_

        # Cari probabilitas untuk label yang diprediksi
        idx = list(classes).index(label)
        probabilitas = float(proba[idx])

        # Buat alasan berdasarkan fitur dominan
        alasan = self._buat_alasan(santri, label, fitur)

        # Snapshot fitur
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

    def _buat_alasan(self, santri: dict, label: str, fitur: np.ndarray) -> str:
        """Buat penjelasan klasifikasi yang mudah dibaca."""
        jilid = int(santri.get("jilid_saat_ini", 0))
        taskih = int(santri.get("total_pengulangan_taskih", 0))

        durasi_diambil = [
            float(santri.get(f"durasi_jilid_{i}", 0) or 0)
            for i in range(7)
            if (santri.get(f"durasi_jilid_{i}") or 0) > 0
        ]
        rata_rata = round(np.mean(durasi_diambil), 1) if durasi_diambil else 0

        jilid_label = "Al-Quran" if jilid == 7 else f"Jilid {jilid}"

        if label == "BBK":
            return (
                f"Santri pada {jilid_label} dengan rata-rata durasi {rata_rata} bulan/jilid "
                f"dan {taskih}x pengulangan taskih dinilai LAYAK naik level (BBK)."
            )
        else:
            return (
                f"Santri pada {jilid_label} dengan rata-rata durasi {rata_rata} bulan/jilid "
                f"dan {taskih}x pengulangan taskih dinilai PERLU pengulangan (TBBK)."
            )

    # ─── Info & Utilities ─────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "is_trained": self.is_trained,
            "versi": self.versi,
            "total_data_latih": self.total_data_latih,
            "aturan_aktif": self.aturan_aktif,
            "feature_names": FEATURE_NAMES,
            "algorithm": "DecisionTreeClassifier (scikit-learn)",
            "params": {
                "max_depth": 8,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "criterion": "gini",
            } if self.clf else {},
        }

    def get_feature_importance(self) -> dict:
        if not self.is_trained:
            raise ValueError("Model belum dilatih")
        return {
            "features": [
                {"nama": name, "importance": round(float(imp), 4)}
                for name, imp in sorted(
                    self.feature_importances_,
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
        }

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump({
            "clf": self.clf,
            "versi": self.versi,
            "total_data_latih": self.total_data_latih,
            "aturan_aktif": self.aturan_aktif,
            "feature_importances_": self.feature_importances_,
        }, MODEL_PATH)
        print(f"  💾 Model disimpan ke {MODEL_PATH}")

    def _load(self):
        try:
            data = joblib.load(MODEL_PATH)
            self.clf = data["clf"]
            self.versi = data["versi"]
            self.total_data_latih = data["total_data_latih"]
            self.aturan_aktif = data.get("aturan_aktif", {})
            self.feature_importances_ = data.get("feature_importances_", [])
            self.is_trained = True
            print(f"  📂 Model dimuat dari disk: {self.versi}")
        except Exception as e:
            print(f"  ⚠️  Gagal load model: {e}")
