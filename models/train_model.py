import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_random_forest(X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test, saved_dir):
    """
    Latih model Random Forest untuk suit dan kategori, lalu simpan model.
    
    Args:
        X_train, X_test: Data fitur yang telah dinormalisasi
        y_suit_train, y_suit_test: Target untuk suit
        y_category_train, y_category_test: Target untuk kategori
        saved_dir (str): Direktori untuk menyimpan model
    
    Returns:
        rf_suit, rf_category: Model Random Forest yang dilatih
    
    Raises:
        ValueError: Jika data masukan tidak valid
        PermissionError: Jika tidak dapat menulis ke direktori
    """
    logger.info("Starting Random Forest training")
    
    # Validasi data masukan
    if X_train.empty or X_test.empty:
        logger.error("Empty training or testing data")
        raise ValueError("Empty training or testing data")
    if len(y_suit_train) == 0 or len(y_category_train) == 0:
        logger.error("Empty target arrays")
        raise ValueError("Empty target arrays")
    
    # Log ukuran dataset
    logger.info(f"Training set size: {len(X_train)} samples")
    logger.info(f"Test set size: {len(X_test)} samples")
    
    # Log distribusi kelas
    logger.info(f"Suit distribution in train: {np.bincount(y_suit_train)}")
    logger.info(f"Suit distribution in test: {np.bincount(y_suit_test)}")
    logger.info(f"Category distribution in train: {np.bincount(y_category_train)}")
    logger.info(f"Category distribution in test: {np.bincount(y_category_test)}")
    
    # Log unique classes
    logger.info(f"Unique suit classes in y_suit_train: {np.unique(y_suit_train)}")
    logger.info(f"Unique category classes in y_category_train: {np.unique(y_category_train)}")
    
    # Model untuk suit
    logger.info("Training Random Forest for suit...")
    rf_suit = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_suit.fit(X_train, y_suit_train)
    
    # Model untuk kategori
    logger.info("Training Random Forest for category...")
    rf_category = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_category.fit(X_train, y_category_train)
    
    # Evaluasi
    logger.info("Evaluating models...")
    suit_pred = rf_suit.predict(X_test)
    category_pred = rf_category.predict(X_test)
    
    # Log unique predicted classes
    logger.info(f"Unique suit classes in suit_pred: {np.unique(suit_pred)}")
    logger.info(f"Unique category classes in category_pred: {np.unique(category_pred)}")
    
    try:
        suit_metrics = precision_recall_fscore_support(y_suit_test, suit_pred, average='macro', zero_division=0)
        category_metrics = precision_recall_fscore_support(y_category_test, category_pred, average='macro', zero_division=0)
        logger.info(f"Suit Metrics (Precision, Recall, F1, Support): {suit_metrics}")
        logger.info(f"Category Metrics (Precision, Recall, F1, Support): {category_metrics}")
        
        # Log confusion matrices
        logger.info(f"Suit Confusion Matrix:\n{confusion_matrix(y_suit_test, suit_pred)}")
        logger.info(f"Category Confusion Matrix:\n{confusion_matrix(y_category_test, category_pred)}")
    except ValueError as e:
        logger.error(f"Evaluation error: {e}")
        raise
    
    ss_accuracy = np.mean(suit_pred == y_suit_test)
    sc_accuracy = np.mean(category_pred == y_category_test)
    cp_accuracy = np.mean((suit_pred == y_suit_test) & (category_pred == y_category_test))
    logger.info(f"SS Accuracy: {ss_accuracy:.3f}, SC Accuracy: {sc_accuracy:.3f}, CP Accuracy: {cp_accuracy:.3f}")
    
    # Simpan model
    try:
        os.makedirs(saved_dir, exist_ok=True)
        joblib.dump(rf_suit, os.path.join(saved_dir, 'rf_suit.pkl'))
        joblib.dump(rf_category, os.path.join(saved_dir, 'rf_category.pkl'))
        logger.info(f"Models saved to {saved_dir}")
    except PermissionError:
        logger.error(f"Cannot write to directory {saved_dir}")
        raise
    
    return rf_suit, rf_category

if __name__ == "__main__":
    # Konfigurasi untuk pengujian langsung
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../data/processed')
    saved_dir = os.path.join(base_dir, 'saved')
    
    try:
        # Baca data hasil preprocessing
        logger.info(f"Loading preprocessed data from {processed_dir}")
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_suit_train = np.load(os.path.join(processed_dir, 'y_suit_train.npy'))
        y_suit_test = np.load(os.path.join(processed_dir, 'y_suit_test.npy'))
        y_category_train = np.load(os.path.join(processed_dir, 'y_category_train.npy'))
        y_category_test = np.load(os.path.join(processed_dir, 'y_category_test.npy'))
        
        # Jalankan pelatihan
        rf_suit, rf_category = train_random_forest(
            X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test, saved_dir
        )
    except FileNotFoundError as e:
        logger.error(f"Preprocessed data not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise