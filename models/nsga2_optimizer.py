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

# Daftar fitur sesuai urutan di CSV hasil preprocess.py
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
        """
        Inisialisasi masalah optimisasi.

        :param feature_vector: list/array dengan 24 fitur numerik hasil ekstraksi
        """
        if len(feature_vector) != len(FEATURE_NAMES):
            raise ValueError(f"feature_vector harus memiliki {len(FEATURE_NAMES)} fitur")

        self.feature_dict = dict(zip(FEATURE_NAMES, feature_vector))
        self.model_dir = os.path.abspath("./models/saved")
        self._load_models()

        # Inisialisasi problem dengan batasan variabel
        super().__init__(
            n_var=5,
            n_obj=1,
            n_constr=0,
            xl=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),   # Lower bounds
            xu=np.array([1.0, 1.0, 1.0, 1.0, 1.0])    # Upper bounds
        )

    def _load_models(self):
        """Muat model dan encoder sekali saja"""
        try:
            self.suit_model = joblib.load(os.path.join(self.model_dir, "rf_contract_suit.pkl"))
            self.suit_encoder = joblib.load(os.path.join(self.model_dir, "label_encoder_suit.pkl"))
            logging.info("Model berhasil dimuat")
        except Exception as e:
            logging.error(f"Gagal memuat model: {e}")
            raise

    def _calculate_hcp_score(self, weight):
        return self.feature_dict["hcp"] * weight

    def _calculate_ltc_score(self, weight):
        return self.feature_dict["ltc"] * (-weight)

    def _calculate_stopper_score(self, weight):
        total = (
            self.feature_dict["stopper_spades"] +
            self.feature_dict["stopper_hearts"] +
            self.feature_dict["stopper_diamonds"] +
            self.feature_dict["stopper_clubs"]
        )
        return total * weight

    def _calculate_distribution_score(self, weight):
        deviation = (
            abs(self.feature_dict["dist_spades"] - 5) +
            abs(self.feature_dict["dist_hearts"] - 5) +
            abs(self.feature_dict["dist_diamonds"] - 5) +
            abs(self.feature_dict["dist_clubs"] - 5)
        )
        return deviation * (-weight)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Fungsi evaluasi untuk NSGA-II.
        x = [weight_hcp, weight_ltc, weight_stopper, weight_distribution, prefer_major]
        """
        weight_hcp, weight_ltc, weight_stopper, weight_distribution, prefer_major = x

        score = 0
        score += self._calculate_hcp_score(weight_hcp)
        score += self._calculate_ltc_score(weight_ltc)
        score += self._calculate_stopper_score(weight_stopper)
        score += self._calculate_distribution_score(weight_distribution)

        # Tambahkan bonus jika suit mayor dipilih dan cocok
        if prefer_major > 0.5:
            suit_pred = self.suit_model.predict(
                pd.DataFrame([self.feature_dict])
            )[0]
            if suit_pred in ['S', 'H']:
                score += 1.0

        # Minimalkan negasi skor
        out["F"] = [-score]


def optimize_contract_strategy(feature_array, n_gen=50):
    """
    Jalankan NSGA-II untuk mencari strategi optimal dalam menentukan kontrak.
    
    :param feature_array: fitur numerik dari tangan North-South (list atau array)
    :param n_gen: jumlah generasi evolusi
    :return: solusi Pareto-optimal (array 2D)
    """
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

    # Pastikan hasil selalu dalam format 2D
    solutions = np.atleast_2d(res.X)

    print("Solusi Pareto-optimal:")
    for i, x in enumerate(solutions[:3]):  # Ambil maksimal 3 strategi
        print(f"Strategi {i+1}: ", {
            "weight_hcp": round(float(x[0]), 2),
            "weight_ltc": round(float(x[1]), 2),
            "weight_stopper": round(float(x[2]), 2),
            "weight_distribution": round(float(x[3]), 2),
            "prefer_major": round(float(x[4]), 2)
        })

    return solutions