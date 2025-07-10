# predict.py

import os
import json
import pandas as pd

from features.extractor import extract_features_from_hand
from models.nsga2_optimizer import optimize_contract_strategy
from utils.recommender import select_best_contract_based_on_all_criteria

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Path model
MODEL_DIR = "./models/saved/"

# Load model dan encoder
try:
    suit_model = joblib.load(os.path.join(MODEL_DIR, "rf_contract_suit.pkl"))
    level_model = joblib.load(os.path.join(MODEL_DIR, "rf_contract_level.pkl"))

    le_suit = joblib.load(os.path.join(MODEL_DIR, "label_encoder_suit.pkl"))
    le_level = joblib.load(os.path.join(MODEL_DIR, "label_encoder_level.pkl"))

except Exception as e:
    raise FileNotFoundError(f"Model atau label encoder tidak ditemukan di {MODEL_DIR}. Jalankan train_model.py terlebih dahulu.")

def predict_contract(hand1, hand2):
    """
    Prediksi kontrak lengkap berdasarkan tangan North-South.
    
    Returns:
    - dict: {
        "final_recommendation": str,
        "valid": bool,
        "confidence_score": float,
        "reasons": list of str,
        "suggestions": list of str
    }
    """
    # 1. Ekstrak fitur dari pasangan tangan
    features = extract_features_from_hand(hand1, hand2, as_dataframe=False)
    
    # 2. Gunakan DataFrame bernama untuk prediksi
    feature_df = pd.DataFrame([features])

    # 3. Prediksi awal dengan Random Forest
    suit_pred_encoded = suit_model.predict(feature_df)[0]
    predicted_suit = le_suit.inverse_transform([suit_pred_encoded])[0]

    level_pred_encoded = level_model.predict(feature_df)[0]
    predicted_level = le_level.inverse_transform([level_pred_encoded])[0]
    predicted_contract = f"{predicted_level}{predicted_suit}"

    # 4. Jalankan NSGA-II untuk strategi alternatif
    feature_array = list(features.values())
    nsga2_solutions = optimize_contract_strategy(feature_array)

    recommendations = []
    for i, x in enumerate(nsga2_solutions[:3]):
        weight_hcp, weight_ltc, weight_stopper, weight_distribution, prefer_major = x
        recommendations.append({
            "strategy_id": i + 1,
            "weight_hcp": round(float(x[0]), 2),
            "weight_ltc": round(float(x[1]), 2),
            "weight_stopper": round(float(x[2]), 2),
            "weight_distribution": round(float(x[3]), 2),
            "prefer_major": round(float(x[4]), 2)
        })

    # 5. Pilih kontrak terbaik berdasarkan validasi aturan bridge dan strategi
    best_recommendation = select_best_contract_based_on_all_criteria(
        features,
        predicted_contract,
        recommendations
    )

    return best_recommendation

if __name__ == "__main__":
    # Contoh input manual
    hand1 = ["AS", "KH", "QD", "JC", "TS", "9H", "8D", "7C", "6S", "5H", "4D", "3C", "2S"]
    hand2 = ["AD", "KD", "QH", "JH", "TH", "9D", "8H", "7D", "6C", "5S", "4H", "3D", "2C"]

    print("=== Sistem Rekomendasi Kontrak Bridge ===")
    result = predict_contract(hand1, hand2)

    print("\n")
    print("=== REKOMENDASI AKHIR ===")
    print(f"Kontrak direkomendasikan: {result['final_recommendation']}")
    print(f"Tingkat keyakinan: {result['confidence_score']:.2f}")

    if result['reasons']:
        print("Alasan:")
        for reason in result['reasons']:
            print(f" - {reason}")

    if result['suggestions']:
        print("Saran:")
        for suggestion in result['suggestions']:
            print(f" - {suggestion}")