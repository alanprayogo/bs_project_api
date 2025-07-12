from utils.validation import validate_contract_based_on_bridge_rules

def select_best_contract_based_on_all_criteria(features, predicted_contract, recommendations, debug=False):
    """
    Pilih kontrak terbaik berdasarkan prediksi ML dan optimisasi NSGA-II.

    Parameters:
    - features (dict): Fitur hasil ekstraksi dari extract_features_from_hand()
    - predicted_contract (str): Kontrak dari model ML (misalnya, "2H")
    - recommendations (list): Daftar bobot dari NSGA-II [[weight_hcp, weight_honor_spades, ...], ...]
    - debug (bool): Jika True, tampilkan informasi debug

    Returns:
    - dict: {
        "final_recommendation": str,
        "valid": bool,
        "confidence_score": float,
        "reasons": list of str,
        "suggestions": list of str
    }
    """
    hcp = features.get("hcp", 0)
    dist_spades = features.get("dist_spades", 0)
    dist_hearts = features.get("dist_hearts", 0)
    dist_diamonds = features.get("dist_diamonds", 0)
    dist_clubs = features.get("dist_clubs", 0)
    balanced_hand1 = features.get("balanced_hand1", 2)
    balanced_hand2 = features.get("balanced_hand2", 2)
    sum_honor_s = features.get("sum_honor_s", 0)
    sum_honor_h = features.get("sum_honor_h", 0)
    sum_honor_d = features.get("sum_honor_d", 0)
    sum_honor_c = features.get("sum_honor_c", 0)
    honor_power = features.get("honor_power", 0)

    # Validasi kontrak ML
    ml_validation = validate_contract_based_on_bridge_rules(features, predicted_contract)
    evaluated_contracts = [{
        "contract": predicted_contract,
        "source": "ML",
        "weights": {},
        "valid": ml_validation["valid"],
        "confidence_score": ml_validation["confidence_score"],
        "reasons": ml_validation["reasons"],
        "suggestions": ml_validation["suggestions"]
    }]

    # Evaluasi kontrak dari NSGA-II
    for rec in recommendations:
        weight_hcp = rec[0]
        weight_honor_spades = rec[1]
        weight_honor_hearts = rec[2]
        weight_honor_diamonds = rec[3]
        weight_honor_clubs = rec[4]
        weight_balance = rec[5]
        weight_suit = rec[6]
        prefer_major = rec[7]

        # Hitung skor HCP dan honor
        hcp_score = hcp * weight_hcp
        honor_score = (
            sum_honor_s * weight_honor_spades +
            sum_honor_h * weight_honor_hearts +
            sum_honor_d * weight_honor_diamonds +
            sum_honor_c * weight_honor_clubs
        )

        # Tentukan level kontrak
        if hcp_score >= 37 and honor_score >= 2.0:
            level = 7  # Grand Slam untuk HCP sangat tinggi
        elif hcp_score >= 32 and honor_score >= 1.5:
            level = 6  # Slam
        else:
            level = max(1, min(7, round(hcp_score / 8 + honor_score / 4)))

        # Tentukan suit
        suit_scores = {}
        if balanced_hand1 in [0, 1] and balanced_hand2 in [0, 1]:
            suit_scores['NT'] = weight_balance * (1 if balanced_hand1 == 0 and balanced_hand2 == 0 else 0.8) + honor_score
        else:
            suit_scores['NT'] = 0  # Tidak mendukung NT jika tangan tidak seimbang

        # Skor suit dengan bonus untuk distribusi panjang
        suit_scores['S'] = (sum_honor_s * weight_honor_spades + weight_suit * dist_spades / 4) * (1.5 if prefer_major > 0.5 else 1.0)
        suit_scores['H'] = (sum_honor_h * weight_honor_hearts + weight_suit * dist_hearts / 4) * (1.5 if prefer_major > 0.5 else 1.0)
        suit_scores['D'] = (sum_honor_d * weight_honor_diamonds + weight_suit * dist_diamonds / 4) * 0.8
        suit_scores['C'] = (sum_honor_c * weight_honor_clubs + weight_suit * dist_clubs / 4) * 0.8
        if max(dist_spades, dist_hearts, dist_diamonds, dist_clubs) >= 10:
            max_suit = max(['S', 'H', 'D', 'C'], key=lambda s: features.get(f'dist_{s.lower()}', 0))
            suit_scores[max_suit] *= 1.3  # Bonus untuk suit terpanjang ≥ 10

        chosen_suit = max(suit_scores, key=suit_scores.get)
        generated_contract = f"{level}{chosen_suit}"
        validation = validate_contract_based_on_bridge_rules(features, generated_contract)

        evaluated_contracts.append({
            "contract": generated_contract,
            "source": "NSGA-II",
            "weights": {
                "hcp": weight_hcp,
                "honor_spades": weight_honor_spades,
                "honor_hearts": weight_honor_hearts,
                "honor_diamonds": weight_honor_diamonds,
                "honor_clubs": weight_honor_clubs,
                "balance": weight_balance,
                "suit": weight_suit,
                "prefer_major": prefer_major
            },
            "valid": validation["valid"],
            "confidence_score": validation["confidence_score"],
            "reasons": validation["reasons"],
            "suggestions": validation["suggestions"]
        })

    # Pilih kontrak terbaik
    best_contract = max(evaluated_contracts, key=lambda x: (
        x["valid"],
        x["confidence_score"],
        1 if x["source"] == "NSGA-II" else 0,  # Prioritaskan NSGA-II
        int(x["contract"][0]) if x["contract"][0].isdigit() else 0,  # Prioritaskan level tinggi
        1 if x["contract"].endswith('NT') else (0.9 if x["contract"].endswith(('S', 'H')) else 0.8)  # Prioritaskan NT, lalu major, lalu minor
    ))

    if debug:
        print("\n[DEBUG] Semua kontrak yang dievaluasi:")
        for contract in evaluated_contracts:
            print(f"Kontrak: {contract['contract']} | Sumber: {contract['source']} | "
                  f"Bobot: {contract['weights']} | Valid: {contract['valid']} | Skor: {contract['confidence_score']}")
            if contract["reasons"]:
                print("Alasan:")
                for reason in contract["reasons"]:
                    print(f" - {reason}")
            if contract["suggestions"]:
                print("Saran:")
                for suggestion in contract["suggestions"]:
                    print(f" - {suggestion}")

    return {
        "final_recommendation": best_contract["contract"],
        "valid": best_contract["valid"],
        "confidence_score": best_contract["confidence_score"],
        "reasons": best_contract["reasons"],
        "suggestions": best_contract["suggestions"]
    }