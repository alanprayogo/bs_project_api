# utils/recommender.py

from .validation import validate_contract_based_on_bridge_rules


def select_best_contract_based_on_all_criteria(features, predicted_contract, nsga2_recommendations):
    """
    Pilih satu kontrak terbaik berdasarkan validasi aturan bridge dan strategi NSGA-II.
    
    Parameters:
    - features (dict): fitur tangan hasil ekstraksi
    - predicted_contract (str): hasil prediksi awal model ML
    - nsga2_recommendations (list of dict): strategi hasil optimisasi
    
    Returns:
    - dict: {
        "final_recommendation": str,
        "confidence_score": float,
        "valid": bool,
        "reasons": list of str,
        "suggestions": list of str
    }
    """
    contract_validations = []

    # Validasi kontrak prediksi awal
    ml_validation = validate_contract_based_on_bridge_rules(features, predicted_contract)
    ml_validation["contract"] = predicted_contract
    ml_validation["source"] = "ml"
    contract_validations.append(ml_validation)

    # Validasi kontrak dari strategi NSGA-II
    for rec in nsga2_recommendations:
        # Bangun kontrak dari bobot strategi
        level_candidates = [int(rec['weight_hcp'] * 7), int(rec['weight_stopper'] * 7)]
        level = max(1, min(7, round(sum(level_candidates) / len(level_candidates)) + 1))
        generated_contract = f"{level}{predicted_contract[1:]}"  # Gunakan suit dari prediksi awal

        val_result = validate_contract_based_on_bridge_rules(features, generated_contract)
        val_result["contract"] = generated_contract
        val_result["source"] = "nsga2"
        contract_validations.append(val_result)

    # Urutkan berdasarkan confidence score
    contract_validations.sort(key=lambda x: x["confidence_score"], reverse=True)

    best = contract_validations[0]

    return {
        "final_recommendation": best["contract"],
        "valid": best["valid"],
        "confidence_score": best["confidence_score"],
        "reasons": best.get("reasons", []),
        "suggestions": best.get("suggestions", [])
    }