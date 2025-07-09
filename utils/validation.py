# utils/validation.py

def validate_contract_based_on_bridge_rules(features, contract):
    """
    Validasi apakah kontrak sesuai dengan aturan bridge.
    
    Parameters:
    - features (dict): fitur hasil ekstraksi dari extract_features_from_hand()
    - contract (str): kontrak seperti "2H", "3NT", dll.

    Returns:
    - dict: {
        "valid": True/False,
        "confidence_score": float,
        "reasons": list of strings,
        "suggestions": list of strings
    }
    """
    level = int(contract[0])
    suit = contract[1:]
    
    hcp_total = features["hcp"]
    ltc = features["ltc"]
    stopper_spades = features["stopper_spades"]
    stopper_hearts = features["stopper_hearts"]
    stopper_diamonds = features["stopper_diamonds"]
    stopper_clubs = features["stopper_clubs"]

    dist_spades = features["dist_spades"]
    dist_hearts = features["dist_hearts"]
    dist_diamonds = features["dist_clubs"]
    dist_clubs = features["dist_clubs"]

    reasons = []
    suggestions = []

    # 1. Validasi HCP vs Level
    if level == 1 and hcp_total < 15:
        reasons.append("Level 1 terlalu rendah untuk HCP sebesar ini")
    elif level == 2 and hcp_total < 20:
        reasons.append("Tidak cukup HCP untuk kontrak level 2+")

    if level >= 3:
        if suit == 'NT':
            if stopper_spades < 2 or stopper_hearts < 2 or stopper_diamonds < 2 or stopper_clubs < 2:
                reasons.append("Tidak memiliki stopper yang cukup untuk kontrak No Trump")
                suggestions.append("Pertimbangkan kontrak suit lain atau raise bidding secara hati-hati")
        elif suit in ['S', 'H']:
            if hcp_total < 24:
                reasons.append("HCP kurang untuk Game di major suit")
        elif suit in ['D', 'C']:
            if hcp_total < 27:
                reasons.append("HCP kurang untuk Game di minor suit")

    # 2. Validasi LTC (Losing Trick Count)
    if suit == 'NT' and ltc > 8:
        reasons.append("LTC terlalu tinggi untuk kontrak No Trump")
        suggestions.append("Pertimbangkan kontrak partial atau cari stopper tambahan")

    if suit in ['S', 'H'] and ltc > 7:
        reasons.append("LTC tinggi untuk major suit game")
        suggestions.append("Pastikan ada distribusi bagus atau stopper kuat")

    # 3. Validasi Slam / Grand Slam
    if level == 6 and hcp_total < 32:
        reasons.append("Tidak cukup HCP untuk Slam")
        suggestions.append("Tingkatkan HCP atau pertimbangkan Game saja")
    elif level == 7 and hcp_total < 37:
        reasons.append("HCP sangat minim untuk Grand Slam")
        suggestions.append("Slam tidak direkomendasikan tanpa HCP ≥37")

    # 4. Preferensi Major Suit (opsional)
    if suit in ['C', 'D'] and hcp_total >= 22 and dist_hearts == 5 or dist_spades == 5:
        suggestions.append("Ada potensi major suit game. Pertimbangkan 4H atau 4S")

    # 5. Ringkasan
    is_valid = len(reasons) == 0
    confidence_score = 1.0 - (len(reasons) * 0.2)

    return {
        "contract": contract,
        "valid": is_valid,
        "confidence_score": max(0.0, min(1.0, confidence_score)),
        "reasons": reasons,
        "suggestions": suggestions
    }