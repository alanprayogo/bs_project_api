from .validation import validate_contract_based_on_bridge_rules
import numpy as np
import pandas as pd

def select_best_contract_based_on_all_criteria(features, predicted_contract, nsga2_recommendations, debug=False):
    """
    Memilih kontrak terbaik berdasarkan fitur, prediksi ML, dan rekomendasi NSGA-II.
    
    Parameters:
    - features: Dict berisi fitur seperti hcp, dist_spades, balanced_hand1, dll.
    - predicted_contract: Kontrak yang diprediksi oleh model ML (format: "3NT", "4S", dll.)
    - nsga2_recommendations: List solusi NSGA-II dengan bobot [weight_hcp, weight_honor_spades, ..., prefer_major]
    - debug: Jika True, cetak informasi debugging
    
    Returns:
    - Dict dengan kontrak terbaik, validitas, skor kepercayaan, alasan, dan saran
    """
    contract_validations = []

    # Evaluasi prediksi awal dari model ML
    ml_validation = validate_contract_based_on_bridge_rules(features, predicted_contract)
    ml_validation["contract"] = predicted_contract
    ml_validation["source"] = "ML"
    contract_validations.append(ml_validation)

    # Evaluasi strategi NSGA-II
    for rec in nsga2_recommendations:
        # Ambil bobot dari solusi NSGA-II
        weight_hcp = rec[0]
        weight_honor_spades = rec[1]
        weight_honor_hearts = rec[2]
        weight_honor_diamonds = rec[3]
        weight_honor_clubs = rec[4]
        weight_balance = rec[5]
        weight_suit = rec[6]
        prefer_major = rec[7]

        # Tentukan level kontrak berdasarkan HCP dan kekuatan honor
        hcp_score = features["hcp"] * weight_hcp
        honor_score = (
            features["sum_honor_s"] * weight_honor_spades +
            features["sum_honor_h"] * weight_honor_hearts +
            features["sum_honor_d"] * weight_honor_diamonds +
            features["sum_honor_c"] * weight_honor_clubs
        )
        level_candidates = [int(hcp_score / 5), int(honor_score / 2)]  # Skala HCP dan honor
        level = max(1, min(7, round(sum(level_candidates) / len(level_candidates))))  # Batas level 1-7

        # Tentukan suit berdasarkan distribusi dan bobot
        suit_scores = {
            'S': (weight_honor_spades * features["sum_honor_s"] + weight_suit * features["dist_spades"] / 4) * (1.5 if prefer_major > 0.5 else 1.0),
            'H': (weight_honor_hearts * features["sum_honor_h"] + weight_suit * features["dist_hearts"] / 4) * (1.5 if prefer_major > 0.5 else 1.0),
            'D': (weight_honor_diamonds * features["sum_honor_d"] + weight_suit * features["dist_diamonds"] / 4) * (0.8 if prefer_major <= 0.5 else 1.0),
            'C': (weight_honor_clubs * features["sum_honor_c"] + weight_suit * features["dist_clubs"] / 4) * (0.8 if prefer_major <= 0.5 else 1.0),
            'NT': (weight_balance * (1 if features["balanced_hand1"] in [0, 1] and features["balanced_hand2"] in [0, 1] else 0) +
                   sum(features[f"sum_honor_{s[0]}"] * rec[i + 1] for i, s in enumerate(['spades', 'hearts', 'diamonds', 'clubs'])))
        }
        
        # Jika kedua tangan seimbang, prioritaskan NT
        if features["balanced_hand1"] in [0, 1] and features["balanced_hand2"] in [0, 1]:
            suit_scores['NT'] *= 1.2  # Bonus untuk NT pada tangan seimbang
        # Jika tangan tidak seimbang, prioritaskan suit terpanjang
        elif features["balanced_hand1"] in [2, 3] or features["balanced_hand2"] in [2, 3]:
            longest_suit = max(
                {'spades': features["dist_spades"], 'hearts': features["dist_hearts"],
                 'diamonds': features["dist_diamonds"], 'clubs': features["dist_clubs"]},
                key=lambda x: features[f"dist_{x}"]
            )
            suit_scores[{'spades': 'S', 'hearts': 'H', 'diamonds': 'D', 'clubs': 'C'}[longest_suit]] *= 1.3

        chosen_suit = max(suit_scores, key=suit_scores.get)
        generated_contract = f"{level}{chosen_suit}"

        # Validasi kontrak
        val_result = validate_contract_based_on_bridge_rules(features, generated_contract)
        val_result["contract"] = generated_contract
        val_result["source"] = "NSGA-II"
        val_result["weights"] = {
            "hcp": round(float(weight_hcp), 2),
            "honor_spades": round(float(weight_honor_spades), 2),
            "honor_hearts": round(float(weight_honor_hearts), 2),
            "honor_diamonds": round(float(weight_honor_diamonds), 2),
            "honor_clubs": round(float(weight_honor_clubs), 2),
            "balance": round(float(weight_balance), 2),
            "suit": round(float(weight_suit), 2),
            "prefer_major": round(float(prefer_major), 2)
        }
        contract_validations.append(val_result)

    # Urutkan berdasarkan confidence_score
    contract_validations.sort(key=lambda x: x["confidence_score"], reverse=True)

    # Debugging
    if debug:
        print("\n[DEBUG] Semua kontrak yang dievaluasi:")
        for c in contract_validations:
            weights = c.get("weights", {})
            weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in weights.items())
            print(f"Kontrak: {c['contract']} | Sumber: {c['source']} | Bobot: {{{weights_str}}} | Valid: {c['valid']} | Skor: {c['confidence_score']:.2f}")

    # Pilih kontrak terbaik
    best = contract_validations[0]
    return {
        "final_recommendation": best["contract"],
        "valid": best["valid"],
        "confidence_score": best["confidence_score"],
        "reasons": best.get("reasons", []),
        "suggestions": best.get("suggestions", [])
    }