# predict.py

import os
import pandas as pd
from features.extractor import extract_features_from_hand
from models.nsga2_optimizer import optimize_contract_strategy
from utils.recommender import select_best_contract_based_on_all_criteria
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

MODEL_DIR = "./models/saved/"

# Load model dan encoder
try:
    suit_model = joblib.load(os.path.join(MODEL_DIR, "rf_contract_suit.pkl"))
    level_model = joblib.load(os.path.join(MODEL_DIR, "rf_contract_level.pkl"))
    le_suit = joblib.load(os.path.join(MODEL_DIR, "label_encoder_suit.pkl"))
    le_level = joblib.load(os.path.join(MODEL_DIR, "label_encoder_level.pkl"))
except Exception as e:
    raise FileNotFoundError(f"Model atau label encoder tidak ditemukan di {MODEL_DIR}. Jalankan train_model.py terlebih dahulu.")

def predict_contract_verbose(hand1, hand2, debug=False):
    print("[1] EKSTRAKSI FITUR DARI TANGAN")
    features = extract_features_from_hand(hand1, hand2, as_dataframe=False)
    feature_df = pd.DataFrame([features])

    hcp_total = features.get("hcp", 0)
    ltc = features.get("ltc", 0)
    dist_spades = features.get("dist_spades", 0)
    dist_hearts = features.get("dist_hearts", 0)
    dist_diamonds = features.get("dist_diamonds", 0)
    dist_clubs = features.get("dist_clubs", 0)

    print(f"- HCP Total: {hcp_total}")
    print(f"- LTC: {ltc}")
    print(f"- Distribusi: {dist_spades}S {dist_hearts}H {dist_diamonds}D {dist_clubs}C")

    # Prediksi awal
    suit_pred_encoded = suit_model.predict(feature_df)[0]
    predicted_suit = le_suit.inverse_transform([suit_pred_encoded])[0]
    level_pred_encoded = level_model.predict(feature_df)[0]
    predicted_level = le_level.inverse_transform([level_pred_encoded])[0]
    predicted_contract = f"{predicted_level}{predicted_suit}"

    print("\n[1] PREDIKSI AWAL DARI MODEL ML")
    print(f"- Prediksi Suit: {predicted_suit}")
    print(f"- Prediksi Level: {predicted_level}")
    print(f"- Kontrak Awal: {predicted_contract}")

    # Optimisasi strategi
    print("\n[2] STRATEGI HASIL OPTIMISASI NSGA-II")
    feature_array = list(features.values())
    nsga2_solutions = optimize_contract_strategy(feature_array, n_gen=100)

    recommendations = []
    for i, x in enumerate(nsga2_solutions[:5]):
        recommendations.append({
            "strategy_id": i + 1,
            "weight_hcp": round(float(x[0]), 2),
            "weight_ltc": round(float(x[1]), 2),
            "weight_stopper": round(float(x[2]), 2),
            "weight_distribution": round(float(x[3]), 2),
            "prefer_major": round(float(x[4]), 2)
        })

    # Pilih kontrak terbaik
    print("\n[3] VALIDASI DAN REKOMENDASI AKHIR")
    best_recommendation = select_best_contract_based_on_all_criteria(
        features,
        predicted_contract,
        recommendations,
        debug=debug
    )

    print(f"- Kontrak Direkomendasikan: {best_recommendation['final_recommendation']}")
    print(f"- Valid: {'Ya' if best_recommendation['valid'] else 'Tidak'}")
    print(f"- Tingkat Keyakinan Akhir: {best_recommendation['confidence_score']:.2f}")

    if best_recommendation['reasons']:
        print("Alasan:")
        for reason in best_recommendation['reasons']:
            print(f" - {reason}")
    if best_recommendation['suggestions']:
        print("Saran:")
        for suggestion in best_recommendation['suggestions']:
            print(f" - {suggestion}")

    return best_recommendation

if __name__ == "__main__":
    hand1 = ["AS", "KS", "QS", "JS", "TS", "9S", "8S", "AH", "KH", "QH", "AD", "KD", "QD"]
    hand2 = ["AC", "KC", "QC", "JC", "TC", "9C", "8C", "7C", "6C", "5C", "4C", "3C", "2C"]

    print("=== SISTEM REKOMENDASI KONTRAK BRIDGE ===\n")
    result = predict_contract_verbose(hand1, hand2, debug=True)