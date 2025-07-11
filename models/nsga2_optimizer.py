# models/nsga2_optimizer.py

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import numpy as np
import joblib
import os
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Daftar fitur sesuai urutan di CSV hasil preprocessing
FEATURE_NAMES = [
    "hcp", "hcp_spades", "hcp_hearts", "hcp_diamonds", "hcp_clubs",
    "dist_spades", "dist_hearts", "dist_diamonds", "dist_clubs",
    "balanced_hand1", "balanced_hand2",
    "stopper_spades", "stopper_hearts", "stopper_diamonds", "stopper_clubs",
    "ltc",
    "num_spades_low", "num_spades_high",
    "num_hearts_low", "num_hearts_high",
    "num_diamonds_low", "num_diamonds_high",
    "num_clubs_low", "num_clubs_high"
]

class ContractOptimizationProblem(ElementwiseProblem):
    def __init__(self, feature_vector):
        # Inisialisasi masalah optimisasi berdasarkan fitur tangan bridge.

        # :param feature_vector: list/array dengan 24 fitur numerik hasil ekstraksi
        if len(feature_vector) != len(FEATURE_NAMES):
            raise ValueError(f"feature_vector harus memiliki {len(FEATURE_NAMES)} fitur")

        # Simpan fitur sebagai dictionary bernama
        self.feature_dict = dict(zip(FEATURE_NAMES, feature_vector))
        self.model_dir = os.path.abspath("./models/saved/")
        self._load_models()

        # Definisi variabel optimisasi
        super().__init__(
            n_var=5,
            n_obj=1,
            n_constr=0,
            xl=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),   # Lower bounds
            xu=np.array([1.0, 1.0, 1.0, 1.0, 1.0])     # Upper bounds
        )

    def _load_models(self):
        # Muat model ML untuk evaluasi strategi
        try:
            self.suit_model = joblib.load(os.path.join(self.model_dir, "rf_contract_suit.pkl"))
            self.level_model = joblib.load(os.path.join(self.model_dir, "rf_contract_level.pkl"))
            self.category_model = joblib.load(os.path.join(self.model_dir, "rf_contract_category.pkl"))

            self.le_suit = joblib.load(os.path.join(self.model_dir, "label_encoder_suit.pkl"))
            self.le_level = joblib.load(os.path.join(self.model_dir, "label_encoder_level.pkl"))
            self.le_category = joblib.load(os.path.join(self.model_dir, "label_encoder_category.pkl"))

            logging.info("Model berhasil dimuat")
        except Exception as e:
            logging.error(f"Gagal memuat model: {e}")
            raise

    def _calculate_hcp_score(self, weight):
        # Semakin tinggi HCP, semakin baik → skor positif
        return self.feature_dict["hcp"] * weight

    def _calculate_ltc_score(self, weight):
        # LTC rendah bagus → jadikan negatif agar minimalkan LTC
        return self.feature_dict["ltc"] * (-weight)

    def _calculate_stopper_score(self, weight):
        # Total stopper semakin tinggi semakin baik
        total = (
            self.feature_dict["stopper_spades"] +
            self.feature_dict["stopper_hearts"] +
            self.feature_dict["stopper_diamonds"] +
            self.feature_dict["stopper_clubs"]
        )
        return total * weight

    def _calculate_distribution_score(self, weight):
        # Distribusi yang tidak rata bisa menjadi keuntungan (tergantung suit)
        deviation = (
            abs(self.feature_dict["dist_spades"] - 5) +
            abs(self.feature_dict["dist_hearts"] - 5) +
            abs(self.feature_dict["dist_diamonds"] - 5) +
            abs(self.feature_dict["dist_clubs"] - 5)
        )
        return deviation * (-weight)

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluasi solusi dalam konteks kontrak bridge.
        # x = [weight_hcp, weight_ltc, weight_stopper, weight_distribution, prefer_major]
        weight_hcp, weight_ltc, weight_stopper, weight_distribution, prefer_major = x

        score = 0
        score += self._calculate_hcp_score(weight_hcp)
        score += self._calculate_ltc_score(weight_ltc)
        score += self._calculate_stopper_score(weight_stopper)
        score += self._calculate_distribution_score(weight_distribution)

        # Tambahkan bonus jika suit mayor (S/H) direkomendasikan oleh model awal
        if prefer_major > 0.5:
            feature_df = pd.DataFrame([self.feature_dict])
            suit_pred_encoded = self.suit_model.predict(feature_df)[0]
            suit_pred = self.le_suit.inverse_transform([suit_pred_encoded])[0]
            if suit_pred in ['S', 'H']:
                score += 1.0

        # Minimalkan negasi skor
        out["F"] = [-score]  # NSGA-II meminimalkan fungsi objektif

def optimize_contract_strategy(feature_array, n_gen=50):
    # Jalankan NSGA-II untuk mencari strategi optimal bidding.
    problem = ContractOptimizationProblem(feature_array)
    algorithm = NSGA2(
        pop_size=100,
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   seed=1,
                   verbose=False)

    solutions = np.atleast_2d(res.X)

    # Cetak hasil terbaik
    print("Solusi Pareto-optimal:")
    for i, x in enumerate(solutions[:3]):
        print(f"Strategi {i+1}: ", {
            "weight_hcp": round(float(x[0]), 2),
            "weight_ltc": round(float(x[1]), 2),
            "weight_stopper": round(float(x[2]), 2),
            "weight_distribution": round(float(x[3]), 2),
            "prefer_major": round(float(x[4]), 2)
        })

    return solutions