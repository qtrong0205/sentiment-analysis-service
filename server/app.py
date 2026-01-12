"""
FastAPI server cho Sentiment Analysis API
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào path để import inference module
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference.predict import SentimentPredictor

# =====================
# APP SETUP
# =====================

app = FastAPI(
    title="Sentiment Analysis API",
    description="API phân tích cảm xúc cho review khách sạn (Việt/Anh)",
    version="1.0.0",
)

# Load model
predictor = SentimentPredictor(
    model_path=str(ROOT_DIR / "final_model")
)


# =====================
# SCHEMAS
# =====================

class InputText(BaseModel):
    text: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Phòng sạch sẽ, nhân viên thân thiện, giá hợp lý."
            }
        }
    }


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
    """Health check endpoint"""
    return {"status": "ok", "message": "Sentiment Analysis API is running"}


@app.post("/predict", response_model=PredictionResult)
def predict(data: InputText):
    """
    Predict sentiment cho một đoạn text
    
    - **text**: Đoạn văn bản cần phân tích (tiếng Việt hoặc tiếng Anh)
    
    Returns:
    - **sentiment**: negative / neutral / positive
    - **score**: 0-100 (0=rất tiêu cực, 100=rất tích cực)
    - **confidence**: độ tin cậy của model
    """
    try:
        result = predictor.predict(data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    """
    Predict sentiment cho nhiều đoạn text cùng lúc
    """
    try:
        results = predictor.predict_batch(data.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# MAIN
# =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
