# utils/validation.py

def validate_contract_based_on_bridge_rules(features, contract):
    """
    Validasi apakah kontrak sesuai dengan aturan bridge berdasarkan:
    - HCP (High Card Points)
    - LTC (Losing Trick Count)
    - Stopper di tiap suit
    - Distribusi tangan
    
    Parameters:
    - features (dict): fitur hasil ekstraksi dari extract_features_from_hand()
    - contract (str): kontrak seperti "2H", "3NT", dll.

    Returns:
    - dict: {
        "contract": str,
        "valid": bool,
        "confidence_score": float,
        "reasons": list of str,
        "suggestions": list of str
    }
    """
    # Ekstrak level dan suit dari kontrak
    try:
        level = int(contract[0])
        suit = contract[1:].upper()
        if len(contract) < 2:
            raise ValueError("Kontrak terlalu pendek")
    except Exception as e:
        raise ValueError(f"Format kontrak tidak valid: '{contract}' → {e}")

    # Ambil nilai penting dari fitur
    hcp_total = features.get("hcp", 0)
    ltc = features.get("ltc", 99)
    stopper_spades = features.get("stopper_spades", 0)
    stopper_hearts = features.get("stopper_hearts", 0)
    stopper_diamonds = features.get("stopper_diamonds", 0)
    stopper_clubs = features.get("stopper_clubs", 0)

    dist_spades = features.get("dist_spades", 0)
    dist_hearts = features.get("dist_hearts", 0)
    dist_diamonds = features.get("dist_diamonds", 0)
    dist_clubs = features.get("dist_clubs", 0)

    reasons = []
    suggestions = []

    # 1. Validasi Umum
    if level == 1 and hcp_total < 15:
        reasons.append("Level 1 terlalu rendah untuk HCP sebesar ini")
    elif level == 2 and hcp_total < 20:
        reasons.append("Tidak cukup HCP untuk kontrak level 2+")

    # 2. Validasi Game Contract (Level ≥3)
    if level >= 3:
        if suit == 'NT':
            if stopper_spades < 2 or stopper_hearts < 2 or stopper_diamonds < 2 or stopper_clubs < 2:
                reasons.append("Tidak memiliki stopper yang cukup untuk kontrak No Trump")
                suggestions.append("Pertimbangkan kontrak suit lain atau raise bidding secara hati-hati")
            if not (dist_spades == 4 and dist_hearts == 4 and dist_diamonds in [2, 3] and dist_clubs in [2, 3]):
                reasons.append("Distribusi tidak ideal untuk kontrak No Trump")
                suggestions.append("Pastikan distribusi kartu seimbang (misalnya 4-4-3-2)")

        elif suit in ['S', 'H']:  # Suit Mayor
            if hcp_total < 24:
                reasons.append("HCP kurang untuk Game di major suit")
                suggestions.append("Pastikan partner memiliki HCP tambahan")
            if dist_spades < 4 and suit == 'S':
                reasons.append("Distribusi Spade tidak mencukupi untuk main di Spade")
            if dist_hearts < 4 and suit == 'H':
                reasons.append("Distribusi Heart tidak mencukupi untuk main di Heart")

        elif suit in ['D', 'C']:  # Suit Minor
            if hcp_total < 27:
                reasons.append("HCP kurang untuk Game di minor suit")
                suggestions.append("Cari partner dengan HCP lebih tinggi atau distribusi bagus")
            if dist_diamonds < 5 and suit == 'D':
                reasons.append("Distribusi Diamond tidak cukup panjang")
            if dist_clubs < 5 and suit == 'C':
                reasons.append("Distribusi Club tidak cukup panjang")

    # 3. Validasi LTC (Losing Trick Count)
    if suit == 'NT' and ltc > 8:
        reasons.append("LTC terlalu tinggi untuk kontrak No Trump")
        suggestions.append("Pertimbangkan kontrak partial atau cari stopper tambahan")
    if suit in ['S', 'H'] and ltc > 7:
        reasons.append("LTC tinggi untuk major suit game")
        suggestions.append("Pastikan ada distribusi bagus atau stopper kuat")
    if suit in ['D', 'C'] and ltc > 6:
        reasons.append("LTC terlalu tinggi untuk minor suit game")

    # 4. Validasi Slam / Grand Slam
    if level == 6 and hcp_total < 32:
        reasons.append("HCP tidak cukup untuk Slam")
        suggestions.append("Tingkatkan HCP atau pertimbangkan Game saja")
    elif level == 7 and hcp_total < 37:
        reasons.append("HCP sangat minim untuk Grand Slam")
        suggestions.append("Slam tidak direkomendasikan tanpa HCP ≥37")

    # 5. Preferensi Major Suit
    if suit in ['C', 'D'] and hcp_total >= 22 and (dist_hearts == 5 or dist_spades == 5):
        suggestions.append("Ada potensi major suit game. Pertimbangkan 4H atau 4S")

    # 6. Ringkasan
    is_valid = len(reasons) == 0
    confidence_score = max(0.0, min(1.0, 1.0 - (len(reasons) * 0.2)))

    return {
        "contract": contract,
        "valid": is_valid,
        "confidence_score": confidence_score,
        "reasons": reasons,
        "suggestions": suggestions
    }