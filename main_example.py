# main.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from src.biding_strategies import BIDING_STRATEGIES
import uvicorn
from typing import List
import re
import os
import subprocess
from pathlib import Path
import json
from predict import predict_contract

import config

app = FastAPI(
    title="Bridge Bidding & Contract Recommendation API",
    description="Sistem rekomendasi biding dan kontrak bridge berbasis ML + NSGA-II + Validasi Aturan Bridge"
)

# ======= Biding =======
class HandRequest(BaseModel):
    cards: List[str]
    strategy: str  # misal: "prec_opening", "sayc_respon_1c"
    
    @validator('cards')
    def validate_card_count(cls, v):
        # Kartu pegangan minimal 13
        if len(v) != 13:
            raise ValueError(f"Jumlah kartu harus 13, ditemukan {len(v)} kartu")
        return v
    
    @validator('cards')
    def validate_no_duplicate_cards(cls, v):
        if len(v) != len(set(v)):
            # Mencari kartu yang duplikat
            seen = set()
            duplicates = []
            for card in v:
                if card in seen:
                    duplicates.append(card)
                else:
                    seen.add(card)
            
            raise ValueError(f"Kartu duplikat ditemukan: {', '.join(duplicates)}")
        return v
    

@app.post("/analisis")
async def analyze_hand(request: HandRequest):
    # Endpoint untuk merekomendasikan kontrak bridge terbaik berdasarkan:
    # - HCP
    # - Distribusi SHDC
    handler = BIDING_STRATEGIES.get(request.strategy)
    if not handler:
        raise HTTPException(
            status_code=400,
            detail=f"Strategi '{request.strategy}' tidak dikenali."
        )

    try:
        return handler(request.cards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======= Kontrak =======
class BridgeHandRequest(BaseModel):
    hand1: list[str]
    hand2: list[str]
    
    @validator('hand1', 'hand2')
    def validate_hand_cards(cls, v, field):
        # Validasi format kartu
        card_pattern = re.compile(r'^[AKQJT2-9][SHDC]$')
        invalid_cards = [card for card in v if not card_pattern.match(card)]
        if invalid_cards:
            raise ValueError(f"Format kartu tidak valid pada {field.name}: {', '.join(invalid_cards)}")
        
        # Validasi jumlah kartu
        if len(v) != 13:
            raise ValueError(f"Jumlah kartu pada {field.name} harus 13, ditemukan {len(v)} kartu")
        
        # Validasi duplikat dalam satu hand
        if len(v) != len(set(v)):
            seen = set()
            duplicates = []
            for card in v:
                if card in seen:
                    duplicates.append(card)
                else:
                    seen.add(card)
            raise ValueError(f"Kartu duplikat dalam {field.name}: {', '.join(duplicates)}")
        
        return v
    
    @validator('hand2')
    def validate_no_duplicate_between_hands(cls, v, values):
        # Validasi duplikat antar hand
        if 'hand1' in values:
            hand1_cards = set(values['hand1'])
            hand2_cards = set(v)
            duplicates = hand1_cards.intersection(hand2_cards)
            if duplicates:
                raise ValueError(f"Kartu duplikat antara hand1 dan hand2: {', '.join(sorted(duplicates))}")
        return v

@app.post("/recommend")
async def recommend_contract(request: BridgeHandRequest):
    try:
        # Run predict_contract from predict.py
        result = predict_contract(request.hand1, request.hand2)
        
        # Format the response to match the terminal output
        response = {
            "result": {
                "early_predicted_contract": result['early_contract'],
                "early_confidence_score": round(result['early_confidence'], 1),
                "predicted_contract": f"{result['level']}{result['suit']}",
                "confidence_score": round(result['confidence'], 1),
                "hand1_hcp": result['hand1_hcp'],
                "hand2_hcp": result['hand2_hcp'],
                "total_hcp": f"{result['total_hcp']} HCP, {result['hcp_strength']} strength",
                "suit_dist": result['suit_dist']
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======= Biding + Deteksi =======
# Lokasi penyimpanan sementara gambar
YOLO_INPUT_PATH = Path.cwd() / "running-yolo" / "images" / "in_biding" / "hand_image.jpg"
# Lokasi biding.py
SCRIPT_PATH = Path.cwd() / "running-yolo" / "biding.py"

print("Current Working Directory:", Path.cwd())
print("YOLO Input Image Path:", YOLO_INPUT_PATH.resolve())

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Baca file dari request
    contents = await file.read()

    # Pastikan folder images/ ada
    YOLO_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Simpan file ke lokasi tetap
    try:
        with open(YOLO_INPUT_PATH, "wb") as f:
            f.write(contents)
        print(f"✅ Gambar berhasil disimpan di {YOLO_INPUT_PATH.resolve()}")
    except Exception as e:
        print(f"❌ Gagal menyimpan gambar: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Gagal menyimpan gambar: {str(e)}"}
        )

    print("🔄 Memulai proses deteksi...")

    # Cek apakah script biding.py tersedia
    if not SCRIPT_PATH.exists():
        print(f"❌ File tidak ditemukan: {SCRIPT_PATH.resolve()}")
        return JSONResponse(
            status_code=500,
            content={"error": f"File biding.py tidak ditemukan di {SCRIPT_PATH.resolve()}"}
        )

    # Jalankan biding.py
    try:
        result = subprocess.run(
            ["python", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            check=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ Gagal menjalankan biding.py")
        print("STDERR:", e.stderr)
        return JSONResponse(
            status_code=500,
            content={"error": "Gagal menjalankan deteksi YOLO"}
        )

    # Baca hasil deteksi
    output_json_path = Path.cwd() / "running-yolo" / "detected_cards.json"
    if not output_json_path.exists():
        return JSONResponse(status_code=500, content={"error": "File hasil deteksi tidak ditemukan"})

    try:
        with open(output_json_path, "r") as f:
            detection_result = json.load(f)
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "File JSON tidak valid"})

    # Kembalikan hasil sebagai JSONResponse
    return {
        "message": "Deteksi selesai",
        "cards": detection_result["detected_cards"]
    }

# ======= Kontrak + Deteksi =======
# Folder penyimpanan sementara
UPLOAD_FOLDER = Path.cwd() / "running-yolo" / "images" / "in_kontrak"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HAND1_PATH = UPLOAD_FOLDER / "hand1.jpg"
HAND2_PATH = UPLOAD_FOLDER / "hand2.jpg"

SCRIPT_PATH = Path.cwd() / "running-yolo" / "kontrak.py"

@app.post("/upload_hand/")
async def upload_hand(file: UploadFile = File(...), hand_number: str = Form('1')):
    HAND_PATH = HAND1_PATH if hand_number == '1' else HAND2_PATH

    try:
        with open(HAND_PATH, "wb") as f:
            f.write(await file.read())
        print(f"✅ Gambar hand{hand_number} berhasil disimpan di {HAND_PATH}")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Gagal menyimpan gambar: {str(e)}"},
        )

    # Jalankan kontrak.py
    print("🔄 Memulai proses deteksi...")
    try:
        result = subprocess.run(
            ["python", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            check=True,
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ Gagal menjalankan kontrak.py")
        print("STDERR:", e.stderr)
        return JSONResponse(
            status_code=500,
            content={"error": "Gagal menjalankan deteksi YOLO"},
        )

    # Baca hasil deteksi
    output_json_path = Path.cwd() / "running-yolo" / "detected_hands.json"
    if not output_json_path.exists():
        return JSONResponse(
            status_code=500,
            content={"error": "File hasil deteksi tidak ditemukan"},
        )

    try:
        with open(output_json_path, "r") as f:
            detection_result = json.load(f)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "File JSON tidak valid"},
        )

    cards_key = "hand1" if hand_number == "1" else "hand2"

    return {
        "message": "Deteksi selesai",
        "cards": detection_result.get(cards_key, []),
    }

if __name__ == "__main__":
    print("Menjalankan API...")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)