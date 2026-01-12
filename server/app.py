"""
FastAPI server cho Sentiment Analysis API
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================
# PATH SETUP
# =====================

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR.parent))

from inference.predict import SentimentPredictor

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "username/model-name"
)

MODEL_DIR = ROOT_DIR / "final_model"

# =====================
# DOWNLOAD MODEL IF NEEDED
# =====================

if not MODEL_DIR.exists():
    AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_DIR))
    AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_DIR)
    )

# =====================
# APP SETUP
# =====================

app = FastAPI(
    title="Sentiment Analysis API",
    description="API phân tích cảm xúc cho review khách sạn (Việt/Anh)",
    version="1.0.0",
)

predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    predictor = SentimentPredictor(model_path=str(MODEL_DIR))

# =====================
# SCHEMAS
# =====================

class InputText(BaseModel):
    text: str

class PredictionResult(BaseModel):
    sentiment: str
    score: int
    confidence: float

class BatchInput(BaseModel):
    texts: list[str]

# =====================
# ENDPOINTS
# =====================

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputText):
    try:
        return predictor.predict(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    try:
        return {"results": predictor.predict_batch(data.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
