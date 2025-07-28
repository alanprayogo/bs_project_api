# preprocess.py

import json
import pandas as pd
from features.extractor import extract_features_from_hand
from tqdm import tqdm
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_dataset(file_path):
    # Membaca dataset JSON bridge.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    logging.info(f"Memuat dataset dari {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def is_valid_hand(hand):
    # Validasi bahwa tangan memiliki 13 kartu unik.
    # Format kartu seperti: ['AS', 'KH', ...]
    if len(hand) != 13:
        return False, "Jumlah kartu tidak 13"
    
    # Cek duplikasi dalam satu hand
    unique_cards = set(hand)
    if len(unique_cards) < 13:
        return False, "Ada duplikasi kartu dalam satu tangan"

    return True, ""

def has_duplicate_cards(hand1, hand2):
    # Cek apakah ada kartu yang muncul di kedua tangan
    duplicates = set(hand1) & set(hand2)
    return len(duplicates) > 0

def create_dataframe_from_dataset(dataset):
    # Buat DataFrame dengan fitur ekstraksi dari tiap board
    rows = []
    for idx, board in enumerate(tqdm(dataset, desc="Mengekstrak Fitur", unit="board")):
        try:
            # Validasi struktur board
            if "hand1" not in board or "hand2" not in board or "contract" not in board:
                logging.warning(f"Board {idx + 1}: Struktur tidak lengkap → dilewati")
                continue

            hand1 = board["hand1"]
            hand2 = board["hand2"]

            # Validasi jumlah kartu
            valid_h1, msg_h1 = is_valid_hand(hand1)
            valid_h2, msg_h2 = is_valid_hand(hand2)

            if not valid_h1:
                logging.warning(f"Board {idx + 1}, hand1 tidak valid: {msg_h1} → dilewati")
                continue
            if not valid_h2:
                logging.warning(f"Board {idx + 1}, hand2 tidak valid: {msg_h2} → dilewati")
                continue

            # Validasi duplikasi kartu antara hand1 dan hand2
            if has_duplicate_cards(hand1, hand2):
                logging.warning(f"Board {idx + 1}: Ada duplikasi kartu antara hand1 dan hand2 → dilewati")
                continue

            # Ekstrak fitur dari pasangan tangan
            features = extract_features_from_hand(hand1, hand2, as_dataframe=False)
            features["contract"] = board["contract"]
            rows.append(features)

        except Exception as e:
            logging.error(f"Error pada board {idx + 1}: {e}")
            continue

    if not rows:
        logging.error("Tidak ada data yang diproses. Dataset tidak valid atau kosong.")
        return None

    df = pd.DataFrame(rows)
    logging.info(f"Ekstraksi selesai. {len(df)} board berhasil diproses.")
    return df


def save_processed_data(df, output_path):
    # Simpan DataFrame ke file CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"\nData berhasil disimpan ke {output_path}")


if __name__ == "__main__":
    # Path input dan output
    input_file = "data/raw/bridge_dataset.json"
    output_file = "data/processed/features.csv"

    try:
        # Baca dataset
        dataset = load_json_dataset(input_file)

        # Proses dataset
        df_features = create_dataframe_from_dataset(dataset)

        if df_features is not None:
            # Simpan hasil ekstraksi
            save_processed_data(df_features, output_file)
        else:
            logging.error("Proses preprocessing gagal karena tidak ada data valid.")

    except Exception as e:
        logging.error(f"Kesalahan fatal saat preprocessing: {e}")