# utils/recommender.py

from .validation import validate_contract_based_on_bridge_rules

def select_best_contract_based_on_all_criteria(features, predicted_contract, nsga2_recommendations, debug=False):
    contract_validations = []

    # Evaluasi prediksi awal
    ml_validation = validate_contract_based_on_bridge_rules(features, predicted_contract)
    ml_validation["contract"] = predicted_contract
    ml_validation["source"] = "ML"
    contract_validations.append(ml_validation)

    # Evaluasi strategi NSGA-II
    for rec in nsga2_recommendations:
        weight_hcp = rec.get("weight_hcp", 0)
        weight_ltc = rec.get("weight_ltc", 0)
        weight_stopper = rec.get("weight_stopper", 0)
        weight_distribution = rec.get("weight_distribution", 0)
        prefer_major = rec.get("prefer_major", 0)

        level_candidates = [int(weight_hcp * 7), int(weight_stopper * 7)]
        level = max(1, min(7, round(sum(level_candidates) / len(level_candidates)) + 1))

        suit_scores = {
            'S': prefer_major * 0.5 + weight_distribution * 0.3 + weight_stopper * 0.2,
            'H': prefer_major * 0.5 + weight_distribution * 0.3 + weight_stopper * 0.2,
            'D': (1 - prefer_major) * 0.4 + weight_distribution * 0.4 + weight_hcp * 0.2,
            'C': (1 - prefer_major) * 0.4 + weight_distribution * 0.4 + weight_hcp * 0.2,
            'NT': (1 - weight_distribution) * 0.6 + weight_stopper * 0.4
        }
        chosen_suit = max(suit_scores, key=suit_scores.get)
        generated_contract = f"{level}{chosen_suit}"

        val_result = validate_contract_based_on_bridge_rules(features, generated_contract)
        val_result["contract"] = generated_contract
        val_result["source"] = "NSGA-II"
        val_result["weights"] = {
            "hcp": round(float(weight_hcp), 2),
            "ltc": round(float(weight_ltc), 2),
            "stopper": round(float(weight_stopper), 2),
            "distribution": round(float(weight_distribution), 2),
            "prefer_major": round(float(prefer_major), 2)
        }
        contract_validations.append(val_result)

    contract_validations.sort(key=lambda x: x["confidence_score"], reverse=True)

    if debug:
        print("\n[DEBUG] Semua kontrak yang dievaluasi:")
        for c in contract_validations:
            weights = c.get("weights", {})
            weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in weights.items())
            print(f"Kontrak: {c['contract']} | Sumber: {c['source']} | Bobot: {{{weights_str}}} | Valid: {c['valid']} | Skor: {c['confidence_score']:.2f}")

    best = contract_validations[0]
    return {
        "final_recommendation": best["contract"],
        "valid": best["valid"],
        "confidence_score": best["confidence_score"],
        "reasons": best.get("reasons", []),
        "suggestions": best.get("suggestions", [])
    }