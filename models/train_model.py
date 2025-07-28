# models/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data(file_path):
    # Muat dataset fitur hasil ekstraksi.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    df = pd.read_csv(file_path)
    return df

def map_contract_to_category(suit, level):
    # Menentukan kategori kontrak berdasarkan suit dan level.
    try:
        level = int(level)
    except ValueError:
        raise ValueError("Level harus berupa angka")

    if level == 7:
        return "Grand Slam"
    elif level == 6:
        return "Slam"
    elif level == 5:
        if suit == "NT" or suit in ["S", "H"] or suit in ["D", "C"]:
            return "Game"
    elif level == 4:
        if suit == "NT":
            return "Game"
        elif suit in ["S", "H"]:
            return "Game"
        else:
            return "Partial"
    elif level == 3:
        if suit == "NT":
            return "Game"
        else:
            return "Partial"
    elif level <= 2:
        return "Partial"
    
    return "Partial"

def extract_suit(contract):
    # Ekstrak suit dari kontrak.
    contract = contract.upper()
    if 'NT' in contract:
        return 'NT'
    elif 'S' in contract:
        return 'S'
    elif 'H' in contract:
        return 'H'
    elif 'D' in contract:
        return 'D'
    elif 'C' in contract:
        return 'C'
    else:
        raise ValueError(f"Invalid suit dalam kontrak: {contract}")

def encode_contract(df):
    # Encode kolom kontrak menjadi suit, level, dan kategori.
    df['contract_suit'] = df['contract'].apply(extract_suit)
    df['contract_level'] = df['contract'].apply(lambda x: x[0])  # Ambil level dari karakter pertama
    df['contract_category'] = df.apply(
        lambda row: map_contract_to_category(row['contract_suit'], row['contract_level']),
        axis=1
    )
    return df

def prepare_features_and_labels(df):
    # Pisahkan fitur dan label.
    feature_columns = [col for col in df.columns if col not in [
        'contract', 'contract_suit', 'contract_level', 'contract_category'
    ]]

    X = df[feature_columns]
    y_suit = df['contract_suit']
    y_level = df['contract_level']
    y_category = df['contract_category']

    return X, y_suit, y_level, y_category

def train_random_forest(X_train, y_train, model_type="suit"):
    # Latih model Random Forest
    print(f"Training model: {model_type}")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder=None):
    # Evaluasi model menggunakan accuracy dan classification report.
    y_pred = model.predict(X_test)
    
    if label_encoder:
        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")
    
    # Tampilkan classification report dengan zero_division=0
    print(classification_report(y_test, y_pred, zero_division=0, digits=3))
    
    return acc

def save_models(suit_model, level_model, category_model,
                le_suit, le_level, le_category,
                output_dir="models/saved/"):
    # Simpan model DAN label encoder ke direktori
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(suit_model, os.path.join(output_dir, "rf_contract_suit.pkl"))
    joblib.dump(level_model, os.path.join(output_dir, "rf_contract_level.pkl"))
    joblib.dump(category_model, os.path.join(output_dir, "rf_contract_category.pkl"))

    joblib.dump(le_suit, os.path.join(output_dir, "label_encoder_suit.pkl"))
    joblib.dump(le_level, os.path.join(output_dir, "label_encoder_level.pkl"))
    joblib.dump(le_category, os.path.join(output_dir, "label_encoder_category.pkl"))

    print("Model dan label encoder berhasil disimpan.")

if __name__ == "__main__":
    DATA_PATH = "./data/processed/features.csv"
    MODEL_DIR = "./models/saved/"

    # 1. Muat dataset
    df = load_data(DATA_PATH)

    if len(df) < 10:
        raise ValueError("Dataset terlalu kecil untuk training. Minimal butuh 10 contoh.")

    # 2. Encode kontrak
    df = encode_contract(df)

    # 3. Siapkan fitur dan label
    X, y_suit, y_level, y_category = prepare_features_and_labels(df)

    # 4. Encode label
    le_suit = LabelEncoder()
    le_level = LabelEncoder()
    le_category = LabelEncoder()

    y_suit_encoded = le_suit.fit_transform(y_suit)
    y_level_encoded = le_level.fit_transform(y_level)
    y_category_encoded = le_category.fit_transform(y_category)

    # 5. Split dataset SEKALI SAJA
    X_train, X_test, y_suit_train, y_suit_test, y_level_train, y_level_test, y_category_train, y_category_test = train_test_split(
        X, y_suit_encoded, y_level_encoded, y_category_encoded,
        test_size=0.3,
        random_state=42
    )

    # 6. Pelatihan model
    print("\nTraining model: Prediksi Suit")
    suit_model = train_random_forest(X_train, y_suit_train, model_type="suit")

    print("\nTraining model: Prediksi Level")
    level_model = train_random_forest(X_train, y_level_train, model_type="level")

    print("\nTraining model: Prediksi Kategori Kontrak")
    category_model = train_random_forest(X_train, y_category_train, model_type="category")

    # 7. Evaluasi model
    print("\nEvaluasi Model: Prediksi Suit")
    evaluate_model(suit_model, X_test, y_suit_test, label_encoder=le_suit)

    print("\nEvaluasi Model: Prediksi Level")
    evaluate_model(level_model, X_test, y_level_test, label_encoder=le_level)

    print("\nEvaluasi Model: Prediksi Kategori Kontrak")
    evaluate_model(category_model, X_test, y_category_test, label_encoder=le_category)

    # 8. Simpan model
    save_models(suit_model, level_model, category_model,
                le_suit, le_level, le_category,
                output_dir=MODEL_DIR)