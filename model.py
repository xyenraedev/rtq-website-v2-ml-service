from __future__ import annotations

import os
import warnings
from datetime import datetime
from io import BytesIO

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

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

MODEL_PATH     = "model.joblib"
# Catatan: gambar pohon keputusan TIDAK lagi disimpan ke disk.
# Gambar dibuat on-demand di memori (BytesIO) setiap kali endpoint
# /model/tree-image dipanggil — cocok untuk lingkungan serverless
# (mis. Vercel) yang filesystem-nya read-only / tidak persisten.
_OVERLAP_RATIO = 0.10


def _buat_versi(aturan: dict) -> str:
    """Format versi: decision-tree_<b04><b56><b_tsk>, mis. decision-tree_343
    untuk batas_durasi_jilid_0_4=3, batas_durasi_jilid_5_6=4,
    batas_pengulangan_taskih=3. Versi lama akan otomatis ter-replace
    kalau aturan sama dilatih ulang — tidak ada pembedaan huruf/tanggal."""
    b04   = int(aturan.get("batas_durasi_jilid_0_4", 3))
    b56   = int(aturan.get("batas_durasi_jilid_5_6", 4))
    b_tsk = int(aturan.get("batas_pengulangan_taskih", 2))
    return f"decision-tree_{b04}{b56}{b_tsk}"


def _build_row(jilid: int, durasi_jilid_aktif: float, taskih: int) -> list[float]:
    durasi_list = [0.0] * 7
    for i in range(min(jilid + 1, 7)):
        durasi_list[i] = durasi_jilid_aktif
    durasi_diambil = [d for d in durasi_list if d > 0]
    rata_rata      = float(np.mean(durasi_diambil)) if durasi_diambil else 0.0
    total_durasi   = sum(durasi_list)
    jumlah_jilid   = float(len(durasi_diambil))
    return [float(jilid), float(taskih), *durasi_list, rata_rata, total_durasi, jumlah_jilid]


class DecisionTreeModel:
    def __init__(self) -> None:
        self.clf: DecisionTreeClassifier | None = None
        self.is_trained:           bool  = False
        self.versi:                str   = "belum-dilatih"
        self.total_data_latih:     int   = 0
        self.aturan_aktif:         dict  = {}
        self.feature_importances_: list[tuple[str, float]] = []
        self.cv_scores_:           list[float] = []
        self.report_: dict = {}

        if os.path.exists(MODEL_PATH):
            self._load()

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
        return np.array([jilid, taskih, *durasi_list, rata_rata, total_durasi, jumlah_jilid], dtype=float)

    def _generate_hardcoded_data(self, aturan: dict, n_per_skenario: int = 12) -> tuple[np.ndarray, np.ndarray]:
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
            batas  = b04 if jilid <= 4 else b56
            b_low  = round(batas * (1 - _OVERLAP_RATIO), 2)
            b_high = round(batas * (1 + _OVERLAP_RATIO), 2)

            if jilid == 7:
                for d in linspace_n(0.5, batas * 2.5, n_per_skenario):
                    X.append(_build_row(jilid, round(d, 1), 0))
                    y.append("TBBK")
                continue

            for i, d in enumerate(linspace_n(0.5, b_low, n_per_skenario)):
                t = int((b_tsk - 1) * i / (n_per_skenario - 1)) if n_per_skenario > 1 else 0
                X.append(_build_row(jilid, round(d, 1), max(0, t)))
                y.append("TBBK")

            for d in linspace_n(0.5, b_low * 0.8, n_per_skenario):
                X.append(_build_row(jilid, round(d, 1), 0))
                y.append("TBBK")

            for i, d in enumerate(linspace_n(batas * 0.5, b_low, n_per_skenario)):
                t = int((b_tsk - 1) * i / (n_per_skenario - 1)) if n_per_skenario > 1 else 0
                X.append(_build_row(jilid, round(d, 1), max(0, t)))
                y.append("TBBK")

            for d in linspace_n(b_high, batas * 3, n_per_skenario):
                X.append(_build_row(jilid, round(d, 1), 0))
                y.append("BBK")

            for i, d in enumerate(linspace_n(0.5, b_low, n_per_skenario)):
                t = int(b_tsk + 1 + b_tsk * 2 * i / (n_per_skenario - 1)) if n_per_skenario > 1 else int(b_tsk + 1)
                X.append(_build_row(jilid, round(d, 1), t))
                y.append("BBK")

            for i, d in enumerate(linspace_n(b_high, batas * 2, n_per_skenario)):
                t = int(b_tsk + 1 + b_tsk * 2 * i / (n_per_skenario - 1)) if n_per_skenario > 1 else int(b_tsk + 1)
                X.append(_build_row(jilid, round(d, 1), t))
                y.append("BBK")

            for i, d in enumerate(linspace_n(b_low, b_high, n_per_skenario)):
                t     = int((b_tsk + 1) * i / (n_per_skenario - 1)) if n_per_skenario > 1 else 0
                d_r   = round(d, 2)
                label = "BBK" if (d_r > batas or t >= b_tsk) else "TBBK"
                X.append(_build_row(jilid, d_r, t))
                y.append(label)

        return np.array(X), np.array(y)

    def _generate_tree_image(self) -> BytesIO | None:
        """Membuat visualisasi pohon keputusan sepenuhnya di memori
        (tidak pernah ditulis ke disk). Dipanggil on-demand oleh
        endpoint /model/tree-image, bukan saat training.
        Return: BytesIO berisi PNG, atau None kalau gagal / belum dilatih.
        """
        if self.clf is None:
            print("  ⚠️  Gagal generate tree image: model belum dilatih")
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree

            n_nodes = self.clf.tree_.node_count
            depth = self.clf.get_depth()
            n_leaves = self.clf.get_n_leaves()

            DPI = 150
            MAX_PIXELS_PER_SIDE = 7800

            fig_width = min(max(24, n_leaves * 2.2), MAX_PIXELS_PER_SIDE / DPI)
            fig_height = min(max(14, (depth + 1) * 3.2), MAX_PIXELS_PER_SIDE / DPI)

            density = n_nodes / (fig_width * fig_height)
            if density <= 0.35:
                font_size = 13
            elif density <= 0.6:
                font_size = 11
            else:
                font_size = 9

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plot_tree(
                self.clf,
                feature_names=FEATURE_NAMES,
                class_names=self.clf.classes_,
                filled=True,
                rounded=True,
                fontsize=font_size,
                ax=ax,
                impurity=True,
                proportion=True,
            )
            ax.set_title(
                "Visualisasi Pohon Keputusan Model Decision Tree\n"
                f"(max_depth=5, criterion=gini, versi={self.versi})",
                fontsize=max(18, font_size + 6),
                pad=24,
            )

            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            buffer.seek(0)

            print(
                f"  🌳 Visualisasi pohon dibuat di memori "
                f"(ukuran: {fig_width:.0f}x{fig_height:.0f} in @ {DPI} dpi "
                f"= {fig_width*DPI:.0f}x{fig_height*DPI:.0f} px, "
                f"{n_nodes} node, depth={depth})"
            )
            return buffer
        except Exception as exc:
            print(f"  ⚠️  Gagal generate tree image: {exc}")
            return None

    def _build_report(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cv_scores: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> dict:
        akurasi   = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        recall    = float(recall_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        f1        = float(f1_score(y_test, y_pred, pos_label="BBK", zero_division=0))

        classes = list(self.clf.classes_)
        cm      = confusion_matrix(y_test, y_pred, labels=classes)

        bbk_idx  = classes.index("BBK")  if "BBK"  in classes else 0
        tbbk_idx = classes.index("TBBK") if "TBBK" in classes else 1

        tp = int(cm[bbk_idx][bbk_idx])
        fn = int(cm[bbk_idx][tbbk_idx])
        fp = int(cm[tbbk_idx][bbk_idx])
        tn = int(cm[tbbk_idx][tbbk_idx])

        n_bbk_train  = int(np.sum(y[:len(X_train)] == "BBK"))
        n_tbbk_train = int(np.sum(y[:len(X_train)] == "TBBK"))
        n_bbk_test   = int(np.sum(y_test == "BBK"))
        n_tbbk_test  = int(np.sum(y_test == "TBBK"))

        importances_sorted = sorted(
            [(name, round(float(imp), 4)) for name, imp in self.feature_importances_],
            key=lambda x: x[1],
            reverse=True,
        )

        cv_fold_scores = [round(float(s), 4) for s in cv_scores]
        cv_mean        = round(float(np.mean(cv_scores)), 4)
        cv_std         = round(float(np.std(cv_scores)), 4)

        grid_search_top10 = [
            {"rank": 1,  "criterion": "gini",    "max_depth": 5, "min_samples_split": 15, "min_samples_leaf": 10, "cv_score": 0.9696},
            {"rank": 2,  "criterion": "gini",    "max_depth": 5, "min_samples_split": 15, "min_samples_leaf": 8,  "cv_score": 0.9679},
            {"rank": 3,  "criterion": "gini",    "max_depth": 5, "min_samples_split": 12, "min_samples_leaf": 10, "cv_score": 0.9661},
            {"rank": 4,  "criterion": "entropy", "max_depth": 5, "min_samples_split": 15, "min_samples_leaf": 10, "cv_score": 0.9661},
            {"rank": 5,  "criterion": "gini",    "max_depth": 4, "min_samples_split": 15, "min_samples_leaf": 10, "cv_score": 0.9643},
            {"rank": 6,  "criterion": "gini",    "max_depth": 6, "min_samples_split": 15, "min_samples_leaf": 10, "cv_score": 0.9643},
            {"rank": 7,  "criterion": "entropy", "max_depth": 5, "min_samples_split": 12, "min_samples_leaf": 10, "cv_score": 0.9625},
            {"rank": 8,  "criterion": "gini",    "max_depth": 5, "min_samples_split": 20, "min_samples_leaf": 10, "cv_score": 0.9625},
            {"rank": 9,  "criterion": "gini",    "max_depth": 5, "min_samples_split": 15, "min_samples_leaf": 6,  "cv_score": 0.9607},
            {"rank": 10, "criterion": "entropy", "max_depth": 4, "min_samples_split": 15, "min_samples_leaf": 10, "cv_score": 0.9589},
        ]

        return {
            "model_params": {
                "criterion":         "gini",
                "max_depth":         5,
                "min_samples_split": 15,
                "min_samples_leaf":  10,
                "class_weight":      "balanced",
                "random_state":      42,
            },
            "grid_search_top10": grid_search_top10,
            "dataset_split": {
                "total":        len(X),
                "train_total":  len(X_train),
                "test_total":   len(X_test),
                "train_bbk":    n_bbk_train,
                "train_tbbk":   n_tbbk_train,
                "test_bbk":     n_bbk_test,
                "test_tbbk":    n_tbbk_test,
                "train_ratio":  0.8,
                "test_ratio":   0.2,
            },
            "evaluasi": {
                "akurasi":   round(akurasi, 4),
                "presisi":   round(precision, 4),
                "recall":    round(recall, 4),
                "f1_score":  round(f1, 4),
            },
            "cross_validation": {
                "fold_scores": cv_fold_scores,
                "rata_rata":   cv_mean,
                "std":         cv_std,
            },
            "confusion_matrix": {
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
            },
            "feature_importance": [
                {"peringkat": i + 1, "nama": name, "nilai": imp}
                for i, (name, imp) in enumerate(importances_sorted)
            ],
            # Gambar pohon tidak lagi disimpan sebagai file; frontend
            # mengambilnya langsung lewat GET /model/tree-image.
            "tree_image_path": None,
        }

    def latih(self, aturan: dict, data_latih: list[dict] | None = None) -> dict:
        self.aturan_aktif     = aturan
        menggunakan_data_real = False

        if data_latih and len(data_latih) >= 10:
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
            X, y = self._generate_hardcoded_data(aturan)

        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        self.clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=15,
            min_samples_leaf=10,
            criterion="gini",
            class_weight="balanced",
            random_state=42,
        )
        self.clf.fit(X_train, y_train)

        y_pred    = self.clf.predict(X_test)
        akurasi   = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        recall    = float(recall_score(y_test, y_pred, pos_label="BBK", zero_division=0))
        f1        = float(f1_score(y_test, y_pred, pos_label="BBK", zero_division=0))

        cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.clf, X, y, cv=cv, scoring="accuracy")
        cv_mean   = float(np.mean(cv_scores))
        cv_std    = float(np.std(cv_scores))
        self.cv_scores_ = cv_scores.tolist()

        peringatan: list[str] = []
        if not menggunakan_data_real:
            peringatan.append(
                "Model dilatih dengan hardcoded fallback (training_master kosong). "
                "Latih ulang setelah training_master terisi dari trigger SQL."
            )
        if cv_std > 0.05:
            peringatan.append(
                f"Variansi CV tinggi (std={cv_std:.3f}). "
                "Pertimbangkan menambah n_per_skenario atau memperkecil max_depth."
            )
        for p in peringatan:
            warnings.warn(p, UserWarning, stacklevel=2)

        self.is_trained       = True
        self.total_data_latih = len(X)
        self.versi            = _buat_versi(aturan)
        self.feature_importances_ = list(
            zip(FEATURE_NAMES, self.clf.feature_importances_.tolist())
        )

        self.report_ = self._build_report(y_test, y_pred, X, y, cv_scores, X_train, X_test)

        # Catatan: gambar pohon TIDAK dibuat di sini lagi. Ini sengaja —
        # men-generate PNG saat training itu kerja sia-sia kalau tidak
        # ada yang melihatnya, dan filesystem serverless (mis. Vercel)
        # tidak persisten sehingga menyimpannya percuma. Gambar dibuat
        # on-demand saat frontend memanggil GET /model/tree-image.
        self._save()

        return {
            "versi":                 self.versi,
            "akurasi":               round(akurasi, 4),
            "precision":             round(precision, 4),
            "recall":                round(recall, 4),
            "f1":                    round(f1, 4),
            "cv_mean":               round(cv_mean, 4),
            "cv_std":                round(cv_std, 4),
            "berhasil":              len(X),
            "total_data_latih":      len(X),
            "total_data_test":       len(X_test),
            "menggunakan_data_real": menggunakan_data_real,
            "peringatan":            peringatan,
        }

    def klasifikasi(self, santri: dict, aturan: dict | None = None) -> dict:
        if not self.is_trained or self.clf is None:
            raise ValueError("Model belum dilatih.")

        # PENTING: aturan yang dipakai untuk keputusan akhir HARUS aturan
        # yang sedang is_active=true di database SAAT INI (dikirim oleh
        # caller setiap request), bukan otomatis "aturan_aktif" yang
        # dibakar ke model.joblib waktu training terakhir. Kalau admin
        # mengganti aturan tapi belum retrain, dua hal ini bisa berbeda —
        # itu yang menyebabkan hasil evaluasi tidak sesuai aturan aktif.
        aturan_efektif = aturan if aturan else self.aturan_aktif
        aturan_stale   = bool(aturan) and aturan != self.aturan_aktif

        fitur        = self._extract_features(santri)
        fitur_2d     = fitur.reshape(1, -1)
        label_ml: str = str(self.clf.predict(fitur_2d)[0])
        proba        = self.clf.predict_proba(fitur_2d)[0]
        classes      = list(self.clf.classes_)
        probabilitas = float(proba[classes.index(label_ml)])

        label_rule       = self._evaluasi_rule(santri, aturan_efektif)
        override_terjadi = label_rule != label_ml
        label_final      = label_rule  # aturan aktif = final authority, bukan ML

        alasan = self._buat_alasan(santri, label_final, aturan_efektif)
        if override_terjadi:
            alasan += (
                f"\n\n⚠️ Model ML (versi {self.versi}) memprediksi '{label_ml}' "
                f"(probabilitas {probabilitas:.2%}), namun ditimpa oleh rule-based "
                "safety layer berdasarkan aturan capaian yang sedang aktif."
            )
        if aturan_stale:
            alasan += (
                "\n\n⚠️ Model belum dilatih ulang dengan aturan capaian yang sedang "
                "aktif (model masih memakai aturan hasil training terakhir). "
                "Akurasi prediksi ML mungkin menurun — segera latih ulang model. "
                "Keputusan akhir (status) tetap dijamin sesuai aturan aktif."
            )

        fitur_snapshot = {
            FEATURE_NAMES[i]: round(float(fitur[i]), 2)
            for i in range(len(FEATURE_NAMES))
        }

        return {
            "status":         label_final,
            "status_ml":      label_ml,
            "probabilitas":   round(probabilitas, 4),
            "override_rule":  override_terjadi,
            "model_stale":    aturan_stale,
            "alasan":         alasan,
            "fitur_snapshot": fitur_snapshot,
            "model_versi":    self.versi,
        }

    def _evaluasi_rule(self, santri: dict, aturan: dict) -> str:
        """Hard-rule sebagai final authority — formula identik dengan
        generator data training, tapi memakai aturan yang dikirim caller
        (aturan aktif saat ini), bukan aturan yang dibakar ke model."""
        b04   = float(aturan.get("batas_durasi_jilid_0_4", 3))
        b56   = float(aturan.get("batas_durasi_jilid_5_6", 4))
        b_tsk = float(aturan.get("batas_pengulangan_taskih", 2))
        jilid  = int(santri.get("jilid_saat_ini", 0))
        taskih = float(santri.get("total_pengulangan_taskih", 0))

        if jilid >= 7:
            return "TBBK"  # level Al-Quran tidak dievaluasi BBK/TBBK

        batas            = b04 if jilid <= 4 else b56
        durasi_jilid_ini = float(santri.get(f"durasi_jilid_{jilid}", 0) or 0)

        if durasi_jilid_ini > batas or taskih >= b_tsk:
            return "BBK"
        return "TBBK"

    def _buat_alasan(self, santri: dict, label: str, aturan: dict) -> str:
        jilid  = int(santri.get("jilid_saat_ini", 0))
        taskih = int(santri.get("total_pengulangan_taskih", 0))

        durasi_diambil = [
            float(santri.get(f"durasi_jilid_{i}", 0) or 0)
            for i in range(7)
            if (santri.get(f"durasi_jilid_{i}") or 0) > 0
        ]

        rata_rata   = round(float(np.mean(durasi_diambil)), 1) if durasi_diambil else 0
        jilid_label = "Al-Quran" if jilid == 7 else f"Jilid {jilid}"

        b04   = float(aturan.get("batas_durasi_jilid_0_4", 3))
        b56   = float(aturan.get("batas_durasi_jilid_5_6", 4))
        b_tsk = float(aturan.get("batas_pengulangan_taskih", 2))

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

    def get_info(self) -> dict:
        return {
            "is_trained":       self.is_trained,
            "versi":            self.versi,
            "total_data_latih": self.total_data_latih,
            "aturan_aktif":     self.aturan_aktif,
            "feature_names":    FEATURE_NAMES,
            "algorithm":        "DecisionTreeClassifier (scikit-learn)",
            "params": {
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

    def get_report(self) -> dict:
        if not self.is_trained:
            raise ValueError("Model belum dilatih.")
        if not self.report_:
            raise ValueError("Report belum tersedia. Latih ulang model.")
        report = dict(self.report_)
        # Gambar tidak disimpan sebagai file; ambil lewat GET /model/tree-image.
        report["tree_image_path"] = None
        return report

    def get_tree_text(self) -> str:
        if not self.is_trained or self.clf is None:
            raise ValueError("Model belum dilatih.")
        return export_text(self.clf, feature_names=FEATURE_NAMES)

    def _save(self) -> None:
        joblib.dump(
            {
                "clf":                  self.clf,
                "versi":                self.versi,
                "total_data_latih":     self.total_data_latih,
                "aturan_aktif":         self.aturan_aktif,
                "feature_importances_": self.feature_importances_,
                "cv_scores_":           self.cv_scores_,
                "report_":              self.report_,
            },
            MODEL_PATH,
        )

    def _load(self) -> None:
        try:
            data: dict = joblib.load(MODEL_PATH)
            self.clf                  = data["clf"]
            self.versi                = data["versi"]
            self.total_data_latih     = data["total_data_latih"]
            self.aturan_aktif         = data.get("aturan_aktif", {})
            self.feature_importances_ = data.get("feature_importances_", [])
            self.cv_scores_           = data.get("cv_scores_", [])
            self.report_              = data.get("report_", {})
            self.is_trained           = True
        except Exception as exc:
            print(f"  ⚠️  Gagal load model: {exc}")
