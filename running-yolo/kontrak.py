from ultralytics import YOLO
import cv2
import os
import json

# Muat model
model = YOLO('../yolo-weights/playingCards.pt')

# Daftar path gambar yang ingin diproses
image_paths = [
    './running-yolo/images/image.png',
    './running-yolo/images/image2.png'
]

# File output JSON
output_file = './running-yolo/detected_hands.json'

# Daftar kelas sesuai model kamu
classNames = ["10C", "10D", "10H", "10S",
              "2C", "2D", "2H", "2S",
              "3C", "3D", "3H", "3S",
              "4C", "4D", "4H", "4S",
              "5C", "5D", "5H", "5S",
              "6C", "6D", "6H", "6S",
              "7C", "7D", "7H", "7S",
              "8C", "8D", "8H", "8S",
              "9C", "9D", "9H", "9S",
              "AC", "AD", "AH", "AS",
              "JC", "JD", "JH", "JS",
              "KC", "KD", "KH", "KS",
              "QC", "QD", "QH", "QS"]

# Prioritas Suit (urutan SHDC)
suit_order = {'S': 0, 'H': 1, 'D': 2, 'C': 3}

# Prioritas Rank
rank_order = {
    'A': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    '10': 9,
    'J': 10,
    'Q': 11,
    'K': 12
}

# Fungsi untuk ekstrak suit dan rank
def card_key(card):
    suit = card[-1]
    rank = card[:-1]
    return (suit_order[suit], rank_order[rank])

# Dictionary untuk menyimpan semua hasil
all_results = {}

# Loop setiap gambar dengan indeks untuk hand1, hand2, dst.
for idx, img_path in enumerate(image_paths, start=1):
    # Baca gambar
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Gambar tidak ditemukan: {img_path}")
        continue

    # Deteksi
    results = model(img)

    detected_classes = []

    # Ekstrak hasil deteksi
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = classNames[cls]
            detected_classes.append(class_name)

    # Hapus duplikasi
    unique_classes = list(set(detected_classes))

    # Urutkan kartu berdasarkan aturan SHDC dan rank
    try:
        sorted_cards = sorted(unique_classes, key=card_key)
    except KeyError as e:
        print("⚠️ Ada format kartu tidak dikenal:", e)
        sorted_cards = unique_classes

    # Simpan ke all_results dengan key 'hand1', 'hand2', dll
    all_results[f"hand{idx}"] = sorted_cards

    print(f"✅ Kartu terdeteksi (hand{idx}):", ', '.join(sorted_cards))

# Simpan semua hasil ke satu file JSON
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"📄 Semua hasil disimpan ke: {output_file}")