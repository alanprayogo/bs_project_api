from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
import logging
from utils.helpers import estimate_score_corrected, map_level_to_category

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BridgeContractProblem(Problem):
    def __init__(self, rf_suit, rf_category, hand_features, scaler, selected_features):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=[0, 1], xu=[4, 7])
        self.rf_suit = rf_suit
        self.rf_category = rf_category
        self.scaler = scaler
        self.selected_features = selected_features
        try:
            self.hand_features = scaler.transform([hand_features])[0][[scaler.feature_names_in_.tolist().index(f) for f in selected_features]]
        except KeyError as e:
            logger.error(f"Feature not found in selected_features: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid hand_features format: {e}")
            raise

    def _evaluate(self, x, out, *args, **kwargs):
        scores = []
        risks = []
        suit_names = {0: 'Spades', 1: 'Hearts', 2: 'Diamonds', 3: 'Clubs', 4: 'No Trump'}
        
        for contract in x:
            try:
                suit, level = int(contract[0]), int(contract[1])
                score = estimate_score_corrected(suit, level)
                suit_prob = self.rf_suit.predict_proba([self.hand_features])[0][suit]
                category = map_level_to_category(level, suit)
                category_prob = self.rf_category.predict_proba([self.hand_features])[0][category]
                risk = 1 - (suit_prob * category_prob)
                total_hcp_raw = self.scaler.inverse_transform([self.hand_features])[0][self.selected_features.index('total_hcp')]
                longest_suit = self.hand_features[self.selected_features.index('longest_suit')]
                # Penalti untuk slam dengan HCP rendah
                if level >= 6 and total_hcp_raw < 30:
                    risk += (30 - total_hcp_raw) * 0.01
                # Penalti untuk suit contract dengan panjang suit pendek
                if suit in [0, 1, 2, 3] and longest_suit < 8:
                    risk += 0.1
                # Bonus untuk suit sangat panjang
                if suit in [0, 1, 2, 3] and longest_suit >= 10:
                    risk = max(0, risk - 0.25)  # Kuat untuk 13 clubs
                # Bonus untuk NT pada HCP sangat tinggi
                if suit == 4 and total_hcp_raw >= 33 and level >= 6:
                    risk = max(0, risk - 0.2)  # Dorong 7NT
                # Fallback untuk probabilitas rendah
                if category_prob < 0.2 and level >= 6 and total_hcp_raw >= 30:
                    risk = max(0, risk - 0.15)  # Kurangi risiko untuk slam
                scores.append(score)
                risks.append(risk)
            except IndexError as e:
                logger.error(f"Invalid suit or category index: {e}")
                raise
            except KeyError as e:
                logger.error(f"Feature not found: {e}")
                raise
        
        out["F"] = np.column_stack([-np.array(scores), np.array(risks)])

def optimize_contract(rf_suit, rf_category, hand_features, scaler, selected_features):
    """
    Jalankan optimasi NSGA-II untuk menemukan kontrak optimal.
    
    Args:
        rf_suit, rf_category: Model Random Forest yang dilatih
        hand_features: Fitur tangan (sebelum normalisasi)
        scaler: Objek StandardScaler
        selected_features: Daftar fitur yang digunakan
    
    Returns:
        best_contract: Kontrak optimal (suit, level)
        confidence: Skor kepercayaan untuk kontrak terpilih
    """
    try:
        problem = BridgeContractProblem(rf_suit, rf_category, hand_features, scaler, selected_features)
        algorithm = NSGA2(pop_size=100, n_gen=50)
        res = minimize(problem, algorithm, ('n_gen', 50), seed=42)
        # Format Pareto front untuk logging
        suit_names = {0: 'Spades', 1: 'Hearts', 2: 'Diamonds', 3: 'Clubs', 4: 'No Trump'}
        pareto_contracts = [(int(x[0]), int(x[1])) for x in res.X]
        pareto_formatted = [f"{level}{suit_names[suit]}" for suit, level in pareto_contracts]
        pareto_scores = res.F[:, 0]
        pareto_risks = res.F[:, 1]
        # Filter Pareto front untuk risiko < 0.9
        valid_indices = [i for i, r in enumerate(pareto_risks) if r < 0.9]
        if not valid_indices:
            valid_indices = range(len(pareto_risks))  # Fallback jika semua risiko tinggi
        # Truncate Pareto front untuk display
        display_limit = 3
        pareto_display = pareto_formatted[:display_limit] + ['...'] if len(pareto_formatted) > display_limit else pareto_formatted
        objectives_display = res.F[:display_limit].tolist() + ['...'] if len(res.F) > display_limit else res.F.tolist()
        logger.info(f"Pareto front contracts: {pareto_display}")
        logger.info(f"Pareto front objectives (score, risk): {objectives_display}")
        # Pilih kontrak dengan keseimbangan skor dan risiko
        total_hcp_raw = scaler.inverse_transform([problem.hand_features])[0][selected_features.index('total_hcp')]
        score_weight = 0.7 if total_hcp_raw >= 25 else 0.3  # Prioritaskan risiko untuk HCP rendah
        risk_weight = 1 - score_weight
        weights = np.array([score_weight, risk_weight])
        normalized_scores = (res.F[valid_indices, 0] - res.F[valid_indices, 0].min()) / (res.F[valid_indices, 0].max() - res.F[valid_indices, 0].min() + 1e-10)
        normalized_risks = (res.F[valid_indices, 1] - res.F[valid_indices, 1].min()) / (res.F[valid_indices, 1].max() - res.F[valid_indices, 1].min() + 1e-10)
        weighted_scores = weights[0] * normalized_scores + weights[1] * normalized_risks
        best_idx = valid_indices[np.argmin(weighted_scores)]
        best_contract = res.X[best_idx]
        # Hitung confidence untuk kontrak terpilih
        suit, level = int(best_contract[0]), int(best_contract[1])
        suit_prob = rf_suit.predict_proba([problem.hand_features])[0][suit]
        category = map_level_to_category(level, suit)
        category_prob = rf_category.predict_proba([problem.hand_features])[0][category]
        confidence = suit_prob * category_prob * 100
        logger.info(f"Selected contract suit_prob: {suit_prob:.3f}, category_prob: {category_prob:.3f}, confidence: {confidence:.1f}%")
        return best_contract, confidence
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise