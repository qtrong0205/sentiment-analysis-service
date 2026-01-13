"""
FastAPI server cho Sentiment Analysis API
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =====================
# PATH SETUP
# =====================

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR.parent))

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "qtrong/sentiment-analysis-model"  # Model trên HuggingFace Hub
)

# =====================
# APP SETUP - Khởi tạo app TRƯỚC khi load model
# =====================

app = FastAPI(
    title="Sentiment Analysis API",
    description="API phân tích cảm xúc cho review khách sạn (Việt/Anh)",
    version="1.0.0",
)

predictor = None

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
    """Health check - trả về ngay lập tức"""
    return {"status": "ok", "message": "Sentiment Analysis API is running"}

@app.get("/health")
def health():
    """Health check for Render"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputText):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please retry in a moment")
    try:
        return predictor.predict(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please retry in a moment")
    try:
        return {"results": predictor.predict_batch(data.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# LOAD MODEL AFTER STARTUP
# =====================

@app.on_event("startup")
async def load_model():
    """Load model sau khi server đã bind port"""
    global predictor
    import asyncio
    
    # Delay nhỏ để đảm bảo server đã sẵn sàng
    await asyncio.sleep(1)
    
    print("🔄 Loading model from HuggingFace...")
    try:
        from inference.predict import SentimentPredictor
        predictor = SentimentPredictor(model_name=MODEL_NAME)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
