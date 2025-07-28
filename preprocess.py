import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import logging
from features.extractor import BridgeHandAnalyzer
from utils.helpers import parse_contract, map_level_to_category

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(json_path, processed_dir, selected_features):
    """
    Preprocess dataset JSON, ekstrak fitur, normalisasi, dan simpan hasilnya.
    
    Args:
        json_path (str): Path ke file JSON dataset.
        processed_dir (str): Direktori untuk menyimpan hasil preprocessing.
        selected_features (list): Daftar 10 fitur utama yang akan digunakan.
    
    Returns:
        X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test, scaler
    
    Raises:
        FileNotFoundError: Jika file JSON tidak ditemukan.
        ValueError: Jika format JSON salah atau ukuran tangan tidak valid.
        KeyError: Jika fitur yang dipilih tidak ada.
        PermissionError: Jika tidak dapat menulis ke direktori.
    """
    logger.info(f"Starting preprocessing with json_path: {json_path}")
    
    analyzer = BridgeHandAnalyzer()
    
    # Baca dataset
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} boards from {json_path}")
    except FileNotFoundError:
        logger.error(f"Dataset not found at {json_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {json_path}")
        raise
    
    features = []
    suits = []
    categories = []
    
    for i, board in enumerate(data):
        try:
            hand1, hand2 = board['hand1'], board['hand2']
            contract = board['contract']
            
            # Validasi ukuran tangan
            if len(hand1) != 13 or len(hand2) != 13:
                raise ValueError(f"Invalid hand size in board {i}: {board}")
            
            # Ekstrak semua fitur
            hand_features = analyzer.extract_comprehensive_features(hand1, hand2)
            
            # Validasi fitur
            missing_features = set(selected_features) - set(hand_features.keys())
            if missing_features:
                raise KeyError(f"Missing features in board {i}: {missing_features}")
            
            # Pilih hanya 10 fitur utama
            selected_hand_features = {k: v for k, v in hand_features.items() if k in selected_features}
            features.append(selected_hand_features)
            
            suit, level = parse_contract(contract)
            category = map_level_to_category(level, suit)
            suits.append(suit)
            categories.append(category)
            
        except KeyError as e:
            logger.error(f"Missing key in board {i}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid data in board {i}: {e}")
            raise
    
    logger.info(f"Extracted features for {len(features)} boards")
    
    X = pd.DataFrame(features)
    y_suit = np.array(suits)
    y_category = np.array(categories)
    
    # Validasi data
    if X.isna().any().any():
        logger.error("NaN values found in features")
        raise ValueError("NaN values found in features")
    
    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    logger.info("Features normalized")
    
    # Bagi data
    X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test = train_test_split(
        X_scaled, y_suit, y_category, test_size=0.2, random_state=42, stratify=y_category
    )
    logger.info(f"Data split: {len(X_train)} training, {len(X_test)} testing")
    
    # Simpan hasil preprocessing
    try:
        os.makedirs(processed_dir, exist_ok=True)
        X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
        np.save(os.path.join(processed_dir, 'y_suit_train.npy'), y_suit_train)
        np.save(os.path.join(processed_dir, 'y_suit_test.npy'), y_suit_test)
        np.save(os.path.join(processed_dir, 'y_category_train.npy'), y_category_train)
        np.save(os.path.join(processed_dir, 'y_category_test.npy'), y_category_test)
        joblib.dump(scaler, os.path.join(processed_dir, 'scaler.pkl'))
        with open(os.path.join(processed_dir, 'selected_features.json'), 'w') as f:
            json.dump(selected_features, f)
        logger.info(f"Preprocessing results saved to {processed_dir}")
    except PermissionError:
        logger.error(f"Cannot write to directory {processed_dir}")
        raise
    
    return X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test, scaler

if __name__ == "__main__":
    # Konfigurasi untuk pengujian langsung
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'data/raw/bridge_dataset.json')
    processed_dir = os.path.join(base_dir, 'data/processed')
    selected_features = [
        'total_hcp', 'dist_spades', 'dist_hearts', 'dist_diamonds', 'dist_clubs',
        'balance_score1', 'balance_score2', 'total_honor_power', 'longest_suit', 'total_controls'
    ]
    
    try:
        preprocess_data(json_path, processed_dir, selected_features)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise