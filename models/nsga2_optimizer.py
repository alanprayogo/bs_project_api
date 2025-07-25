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
    "sum_honor_s", "sum_honor_h", "sum_honor_d", "sum_honor_c", "honor_power",
    "num_spades_low", "num_spades_high",
    "num_hearts_low", "num_hearts_high",
    "num_diamonds_low", "num_diamonds_high",
    "num_clubs_low", "num_clubs_high"
]

class ContractOptimizationProblem(ElementwiseProblem):
    def __init__(self, feature_vector):
        if len(feature_vector) != len(FEATURE_NAMES):
            raise ValueError(f"feature_vector harus memiliki {len(FEATURE_NAMES)} fitur")

        self.feature_dict = dict(zip(FEATURE_NAMES, feature_vector))
        self.model_dir = os.path.abspath("./models/saved/")
        self._load_models()

        super().__init__(
            n_var=8,  # HCP, honor_spades, honor_hearts, honor_diamonds, honor_clubs, balance, suit, prefer_major
            n_obj=2,  # Objektif: skor total dan kecocokan suit
            n_constr=0,
            xl=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            xu=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )

    def _load_models(self):
        try:
            self.suit_model = joblib.load(os.path.join(self.model_dir, "rf_contract_suit.pkl"))
            self.level_model = joblib.load(os.path.join(self.model_dir, "rf_contract_level.pkl"))
            self.le_suit = joblib.load(os.path.join(self.model_dir, "label_encoder_suit.pkl"))
            self.le_level = joblib.load(os.path.join(self.model_dir, "label_encoder_level.pkl"))
            logging.info("Model berhasil dimuat")
        except Exception as e:
            logging.error(f"Gagal memuat model: {e}")
            raise

    def _calculate_hcp_score(self, weight): 
        return self.feature_dict["hcp"] * weight

    def _calculate_honor_suit_score(self, weight_spades, weight_hearts, weight_diamonds, weight_clubs):
        scores = {
            'spades': self.feature_dict["sum_honor_s"] * weight_spades * max(1, self.feature_dict["dist_spades"] / 4),
            'hearts': self.feature_dict["sum_honor_h"] * weight_hearts * max(1, self.feature_dict["dist_hearts"] / 4),
            'diamonds': self.feature_dict["sum_honor_d"] * weight_diamonds * max(1, self.feature_dict["dist_diamonds"] / 4),
            'clubs': self.feature_dict["sum_honor_c"] * weight_clubs * max(1, self.feature_dict["dist_clubs"] / 4)
        }
        return scores

    def _calculate_distribution_score(self, weight_balance, weight_suit):
        # Prediksi suit dari model
        feature_df = pd.DataFrame([self.feature_dict])
        suit_pred_encoded = self.suit_model.predict(feature_df)[0]
        suit_pred = self.le_suit.inverse_transform([suit_pred_encoded])[0]
        
        # Keseimbangan tangan untuk no-trump
        balance_score = 0
        if self.feature_dict["balanced_hand1"] in [0, 1] and self.feature_dict["balanced_hand2"] in [0, 1]:
            if suit_pred == 'NT':
                # Hitung jumlah suit dengan honor kuat (sum_honor >= 1.0)
                strong_suits = sum(1 for s in ['spades', 'hearts', 'diamonds', 'clubs'] if self.feature_dict[f"sum_honor_{s[0]}"] >= 1.0)
                balance_score += 0.5 + 0.1 * strong_suits  # Bonus tambahan untuk honor kuat
            else:
                balance_score += 0.2  # Bonus kecil untuk suit kontrak
        
        # Preferensi suit untuk tangan tidak seimbang
        suit_score = 0
        if self.feature_dict["balanced_hand1"] in [2, 3] or self.feature_dict["balanced_hand2"] in [2, 3]:
            # Temukan suit terpanjang
            suits = {
                'spades': self.feature_dict["dist_spades"],
                'hearts': self.feature_dict["dist_hearts"],
                'diamonds': self.feature_dict["dist_diamonds"],
                'clubs': self.feature_dict["dist_clubs"]
            }
            longest_suit = max(suits, key=suits.get)
            longest_count = suits[longest_suit]
            
            # Bonus untuk suit terpanjang, lebih besar untuk major suit
            if longest_count >= 8:
                suit_score += 1.0 if longest_suit in ['spades', 'hearts'] else 0.8
            elif longest_count >= 6:
                suit_score += 0.5 if longest_suit in ['spades', 'hearts'] else 0.4
            
            # Bonus tambahan jika suit terpanjang sesuai dengan prediksi
            if suit_pred in ['S', 'H', 'D', 'C'] and longest_suit == {'S': 'spades', 'H': 'hearts', 'D': 'diamonds', 'C': 'clubs'}[suit_pred]:
                suit_score += 0.3

        return balance_score * weight_balance, suit_score * weight_suit

    def _evaluate(self, x, out, *args, **kwargs):
        weight_hcp, weight_honor_spades, weight_honor_hearts, weight_honor_diamonds, weight_honor_clubs, weight_balance, weight_suit, prefer_major = x
        
        # Skor total untuk kekuatan tangan
        total_score = 0
        total_score += self._calculate_hcp_score(weight_hcp)
        
        # Skor distribusi
        balance_score, suit_score = self._calculate_distribution_score(weight_balance, weight_suit)
        
        # Prediksi suit dari model
        feature_df = pd.DataFrame([self.feature_dict])
        suit_pred_encoded = self.suit_model.predict(feature_df)[0]
        suit_pred = self.le_suit.inverse_transform([suit_pred_encoded])[0]
        
        # Skor honor per suit
        honor_scores = self._calculate_honor_suit_score(weight_honor_spades, weight_honor_hearts, weight_honor_diamonds, weight_honor_clubs)
        
        # Skor kecocokan suit
        if suit_pred == 'NT':
            # Untuk NT, gunakan balance_score dan jumlah honor kuat per suit
            total_score += balance_score
            strong_suits = sum(1 for s in ['spades', 'diamonds', 'clubs'] if self.feature_dict[f"sum_honor_{s[0]}"] >= 1.0)
            suit_score = sum(honor_scores[s] for s in ['spades', 'hearts', 'diamonds', 'clubs'] if self.feature_dict[f"sum_honor_{s[0]}"] >= 1.0) * (0.5 + 0.1 * strong_suits)
        else:
            # Untuk suit kontrak, gunakan suit_score dan honor suit yang diprediksi
            total_score += suit_score
            suit_score = honor_scores.get({'S': 'spades', 'H': 'hearts', 'D': 'diamonds', 'C': 'clubs'}[suit_pred], 0)
            if prefer_major > 0.5 and suit_pred in ['S', 'H']:
                suit_score *= 1.5  # Bonus untuk major suit

        # Bonus untuk tangan kuat (mendukung Slam/Grand Slam)
        if self.feature_dict["hcp"] >= 26 and self.feature_dict["honor_power"] >= 7:
            total_score += 0.4

        # Output: dua objektif (skor total dan kecocokan suit)
        out["F"] = [-total_score, -suit_score]

def optimize_contract_strategy(feature_array, n_gen=50):
    problem = ContractOptimizationProblem(feature_array)
    algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=False)
    solutions = np.atleast_2d(res.X)
    return solutions