import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from features.extractor import BridgeHandAnalyzer
from models.nsga2_optimizer import optimize_contract

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_hcp(hand):
    """Calculate High Card Points (HCP) for a hand."""
    hcp = 0
    for card in hand:
        rank = card[0] if len(card) == 3 else card[:1]
        if rank == 'A':
            hcp += 4
        elif rank == 'K':
            hcp += 3
        elif rank == 'Q':
            hcp += 1
        elif rank == 'J':
            hcp += 1
        elif rank == 'T':
            hcp += 1
    return hcp

def get_suit_distribution(hand1, hand2):
    """Get combined suit distribution strength."""
    suits = {'S': 0, 'H': 0, 'D': 0, 'C': 0}
    for card in hand1 + hand2:
        suit = card[-1]
        suits[suit] += 1
    dist = []
    for suit, count in suits.items():
        if count >= 8:
            dist.append(f"strong {suit}")
        elif count >= 5:
            dist.append(f"moderate {suit}")
    return ", ".join(dist) if dist else "balanced"

def map_category_to_level(category):
    """Map category to contract level."""
    category_to_level = {0: 3, 1: 5, 2: 6, 3: 7}  # Partial game: 3, Game: 5, Small slam: 6, Grand slam: 7
    return category_to_level.get(category, 3)

def predict_contract(hand1, hand2):
    """
    Prediksi kontrak optimal untuk dua tangan bridge menggunakan model yang sudah dilatih.
    
    Args:
        hand1 (list): Daftar 13 kartu untuk tangan pertama.
        hand2 (list): Daftar 13 kartu untuk tangan kedua.
    
    Returns:
        dict: Kontrak optimal dengan kunci 'suit', 'level', 'confidence', serta informasi HCP, distribusi suit, dan early prediction.
    
    Raises:
        FileNotFoundError: Jika file model atau scaler tidak ditemukan.
        ValueError: Jika input tangan tidak valid atau fitur tidak sesuai.
    """
    logger.info("Starting contract prediction")
    
    # Validasi input
    if len(hand1) != 13 or len(hand2) != 13:
        logger.error("Each hand must contain exactly 13 cards")
        raise ValueError("Each hand must contain exactly 13 cards")
    
    # Tentukan direktori
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'data/processed')
    saved_dir = os.path.join(base_dir, 'models/saved')
    
    # Muat model, scaler, dan selected_features
    try:
        rf_suit = joblib.load(os.path.join(saved_dir, 'rf_suit.pkl'))
        rf_category = joblib.load(os.path.join(saved_dir, 'rf_category.pkl'))
        scaler = joblib.load(os.path.join(processed_dir, 'scaler.pkl'))
        with open(os.path.join(processed_dir, 'selected_features.json'), 'r') as f:
            selected_features = json.load(f)
        logger.info("Loaded models, scaler, and selected features")
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise
    
    # Ekstrak fitur
    try:
        analyzer = BridgeHandAnalyzer()
        hand_features = analyzer.extract_comprehensive_features(hand1, hand2)
        selected_hand_features = [hand_features[f] for f in selected_features]
        scaled_features = scaler.transform([selected_hand_features])[0]
        logger.info("Extracted features for the hand")
        
        # Hitung HCP dan distribusi suit
        hand1_hcp = calculate_hcp(hand1)
        hand2_hcp = calculate_hcp(hand2)
        total_hcp = hand1_hcp + hand2_hcp
        hcp_strength = "moderate" if total_hcp < 25 else "strong"
        suit_dist = get_suit_distribution(hand1, hand2)
        
        logger.info(f"hand1_hcp: {hand1_hcp}")
        logger.info(f"hand2_hcp: {hand2_hcp}")
        logger.info(f"total_hcp: {total_hcp} HCP, {hcp_strength} strength")
        logger.info(f"suit_dist: {suit_dist}")
        
        # Early prediction dari Random Forest
        suit_names = {0: 'Spades', 1: 'Hearts', 2: 'Diamonds', 3: 'Clubs', 4: 'No Trump'}
        suit_abbr = {0: 'S', 1: 'H', 2: 'D', 3: 'C', 4: 'NT'}
        early_suit = rf_suit.predict([scaled_features])[0]
        early_category = rf_category.predict([scaled_features])[0]
        early_level = map_category_to_level(early_category)
        early_suit_prob = rf_suit.predict_proba([scaled_features])[0][early_suit]
        early_category_prob = rf_category.predict_proba([scaled_features])[0][early_category]
        early_confidence = early_suit_prob * early_category_prob * 100
        early_contract = f"{early_level}{suit_names[early_suit]}"
        early_contract_abbr = f"{early_level}{suit_abbr[early_suit]}"
        logger.info(f"Early predicted contract: {early_contract}, confidence: {early_confidence:.1f}%")
    except KeyError as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
    
    # Optimasi kontrak
    try:
        best_contract, confidence = optimize_contract(rf_suit, rf_category, selected_hand_features, scaler, selected_features)
        suit, level = int(best_contract[0]), int(best_contract[1])
        logger.info(f"Optimal contract: {level}{suit_names[suit]}")
        
        return {
            'suit': suit_abbr[suit],
            'level': level,
            'confidence': confidence,
            'hand1_hcp': hand1_hcp,
            'hand2_hcp': hand2_hcp,
            'total_hcp': total_hcp,
            'hcp_strength': hcp_strength,
            'suit_dist': suit_dist,
            'early_contract': early_contract_abbr,
            'early_confidence': early_confidence
        }
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    # Contoh tangan untuk pengujian
    # hand1 = ["AS", "KS", "QS", "JS", "TS", "9S", "8S", "AH", "KH", "QH", "AD", "KD", "QD"]
    # hand2 = ["AC", "KC", "QC", "JC", "TC", "9C", "8C", "7C", "6C", "5C", "4C", "3C", "2C"]

    hand1 = ["TS", "KH", "TH", "6H", "5H", "4H", "JD", "9D", "8D", "5D", "QC", "9C", "8C"]
    hand2 = ["KS", "JS", "8S", "7S", "6S", "4S", "2S", "AH", "3H", "4D", "AC", "7C", "3C"]
    
    try:
        result = predict_contract(hand1, hand2)
        print("\nResult:")
        print(f"Early predicted contract: {result['early_contract']}, Confidence Score: {result['early_confidence']:.1f}%")
        print(f"Predicted contract: {result['level']}{result['suit']}, Confidence Score: {result['confidence']:.1f}%")
        print(f"hand1_hcp: {result['hand1_hcp']}")
        print(f"hand2_hcp: {result['hand2_hcp']}")
        print(f"total_hcp: {result['total_hcp']} HCP, {result['hcp_strength']} strength")
        print(f"suit_dist: {result['suit_dist']}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise