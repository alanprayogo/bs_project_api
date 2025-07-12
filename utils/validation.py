def validate_contract_based_on_bridge_rules(features, contract):
    """
    Validasi apakah kontrak sesuai dengan aturan bridge berdasarkan:
    - HCP (High Card Points)
    - Honor per suit (sum_honor_s, sum_honor_h, sum_honor_d, sum_honor_c)
    - Distribusi tangan (dist_spades, dist_hearts, dist_diamonds, dist_clubs)
    - Keseimbangan tangan (balanced_hand1, balanced_hand2)
    
    Parameters:
    - features (dict): Fitur hasil ekstraksi dari extract_features_from_hand()
    - contract (str): Kontrak seperti "2H", "3NT", dll.

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
        if len(contract) < 2 or suit not in ['S', 'H', 'D', 'C', 'NT']:
            raise ValueError("Kontrak tidak valid")
    except Exception as e:
        return {
            "contract": contract,
            "valid": False,
            "confidence_score": 0.0,
            "reasons": [f"Format kontrak tidak valid: '{contract}' → {e}"],
            "suggestions": []
        }

    # Ambil nilai penting dari fitur
    hcp_total = features.get("hcp", 0)
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

    reasons = []
    suggestions = []

    # 1. Validasi Umum
    if level == 1 and hcp_total < 15:
        reasons.append("Level 1 terlalu rendah untuk HCP < 15")
        suggestions.append("Pertimbangkan pass atau kontrak level 2 dengan partner yang kuat")
    elif level == 2 and hcp_total < 20:
        reasons.append("HCP kurang untuk kontrak level 2")
        suggestions.append("Periksa distribusi atau dukungan dari partner")

    # 2. Validasi Game Contract (Level ≥ 3)
    if level >= 3:
        if suit == 'NT':
            if level not in [3, 4, 5]:
                reasons.append("Level tidak valid untuk Game No-Trump (harus 3, 4, atau 5)")
                suggestions.append("Pertimbangkan level 3–5 untuk No-Trump")
            if hcp_total < 24:
                reasons.append("HCP kurang untuk Game No-Trump (minimal 24)")
                suggestions.append("Pertimbangkan kontrak partial atau suit")
            if balanced_hand1 not in [0, 1] or balanced_hand2 not in [0, 1]:
                reasons.append("Tangan tidak seimbang untuk kontrak No-Trump")
                suggestions.append("Pertimbangkan kontrak suit dengan distribusi panjang")
            # Periksa honor kuat di setidaknya 3 suit
            strong_suits = sum(1 for s in [sum_honor_s, sum_honor_h, sum_honor_d, sum_honor_c] if s >= 1.0)
            if strong_suits < 3:
                reasons.append(f"Hanya {strong_suits} suit dengan honor kuat (sum_honor >= 1.0) untuk No-Trump")
                suggestions.append("Pastikan honor kuat di minimal 3 suit")

        elif suit in ['S', 'H']:  # Suit Mayor
            if level not in [4, 5]:
                reasons.append("Level tidak valid untuk Game di major suit (harus 4 atau 5)")
                suggestions.append("Pertimbangkan level 4 atau 5 untuk major suit")
            if hcp_total < 24:
                reasons.append("HCP kurang untuk Game di major suit (minimal 24)")
                suggestions.append("Pastikan partner memiliki HCP tambahan")
            if suit == 'S' and dist_spades < 8:
                reasons.append("Distribusi Spades kurang dari 8 kartu untuk dua tangan")
                suggestions.append("Pertimbangkan suit lain atau No-Trump jika seimbang")
            if suit == 'H' and dist_hearts < 8:
                reasons.append("Distribusi Hearts kurang dari 8 kartu untuk dua tangan")
                suggestions.append("Pertimbangkan suit lain atau No-Trump jika seimbang")
            if suit == 'S' and sum_honor_s < 1.0:
                reasons.append("Honor Spades lemah (sum_honor_s < 1.0)")
                suggestions.append("Pertimbangkan suit dengan honor lebih kuat")
            if suit == 'H' and sum_honor_h < 1.0:
                reasons.append("Honor Hearts lemah (sum_honor_h < 1.0)")
                suggestions.append("Pertimbangkan suit dengan honor lebih kuat")

        elif suit in ['D', 'C']:  # Suit Minor
            if level != 5:
                reasons.append("Level tidak valid untuk Game di minor suit (harus 5)")
                suggestions.append("Pertimbangkan level 5 untuk minor suit")
            if hcp_total < 27:
                reasons.append("HCP kurang untuk Game di minor suit (minimal 27)")
                suggestions.append("Pertimbangkan major suit atau No-Trump jika seimbang")
            if suit == 'D' and dist_diamonds < 8:
                reasons.append("Distribusi Diamonds kurang dari 8 kartu untuk dua tangan")
                suggestions.append("Pertimbangkan suit lain dengan distribusi lebih panjang")
            if suit == 'C' and dist_clubs < 8:
                reasons.append("Distribusi Clubs kurang dari 8 kartu untuk dua tangan")
                suggestions.append("Pertimbangkan suit lain dengan distribusi lebih panjang")
            if suit == 'D' and sum_honor_d < 1.0:
                reasons.append("Honor Diamonds lemah (sum_honor_d < 1.0)")
                suggestions.append("Pertimbangkan suit dengan honor lebih kuat")
            if suit == 'C' and sum_honor_c < 1.0:
                reasons.append("Honor Clubs lemah (sum_honor_c < 1.0)")
                suggestions.append("Pertimbangkan suit dengan honor lebih kuat")

    # 3. Validasi Slam / Grand Slam
    if level == 6:
        if hcp_total < 32:
            reasons.append("HCP kurang untuk Slam (minimal 32)")
            suggestions.append("Pertimbangkan Game (level 3–5)")
        if honor_power < 7:
            reasons.append("Honor power terlalu rendah untuk Slam (minimal 7)")
            suggestions.append("Pastikan memiliki kartu honor kuat di beberapa suit")
    elif level == 7:
        if hcp_total < 37:
            reasons.append("HCP kurang untuk Grand Slam (minimal 37)")
            suggestions.append("Pertimbangkan Slam (level 6) atau Game")
        if honor_power < 8:
            reasons.append("Honor power terlalu rendah untuk Grand Slam (minimal 8)")
            suggestions.append("Pastikan memiliki kartu honor kuat di semua suit")

    # 4. Preferensi Major Suit
    if suit in ['C', 'D'] and hcp_total >= 22 and (dist_hearts >= 8 or dist_spades >= 8):
        suggestions.append("Potensi major suit game (4H atau 4S) dengan distribusi 8+ kartu")
    if suit == 'NT' and (dist_hearts >= 8 or dist_spades >= 8) and (sum_honor_h >= 1.0 or sum_honor_s >= 1.0):
        suggestions.append("Pertimbangkan major suit (4H atau 4S) karena distribusi panjang dan honor kuat")

    # 5. Ringkasan
    is_valid = len(reasons) == 0
    confidence_score = max(0.0, min(1.0, 1.0 - (len(reasons) * 0.15)))

    return {
        "contract": contract,
        "valid": is_valid,
        "confidence_score": confidence_score,
        "reasons": reasons,
        "suggestions": suggestions
    }