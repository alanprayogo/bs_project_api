# bid_snapper_backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.biding_strategies import BIDING_STRATEGIES

app = FastAPI()

class HandRequest(BaseModel):
    cards: list[str]
    strategy: str  # misal: "prec_opening", "sayc_respon_1c"

@app.post("/analisis")
async def analyze_hand(request: HandRequest):
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