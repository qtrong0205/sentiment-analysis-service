"""
Module load model và predict sentiment
Dùng cho FastAPI / Backend service
"""

import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import Optional, List

# =====================
# CONFIG
# =====================

MODEL_PATH = "../final_model"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

# =====================
# PREDICTOR CLASS
# =====================

class SentimentPredictor:
    def __init__(self, model_path: str = None, model_name: str = None):
        """
        Khởi tạo predictor
        - model_path: đường dẫn local tới model
        - model_name: tên model trên HuggingFace Hub (vd: "username/model-name")
        """
        # Ưu tiên model_name (HuggingFace) nếu được cung cấp
        source = model_name if model_name else (model_path if model_path else MODEL_PATH)
        local_only = model_name is None and model_path is not None
        
        print(f"Loading model from: {source} (local_only={local_only})")
        if HF_TOKEN:
            print("Using HF_TOKEN for authentication")

        tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=local_only,
            token=HF_TOKEN
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            source,
            local_files_only=local_only,
            token=HF_TOKEN
        )

        # Sử dụng device=-1 cho CPU (tương thích với Render free tier)
        self.classifier = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1  # -1 = CPU, 0 = GPU
        )

    @staticmethod
    def normalize_score(sentiment: str, confidence: float) -> int:
        """
        Chuyển đổi sentiment + confidence thành score 0–100
        """
        if sentiment == "positive":
            return int(50 + confidence * 50)
        if sentiment == "negative":
            return int(50 - confidence * 50)
        return 50

    def predict(self, text: str) -> dict:
        """
        Predict sentiment cho một đoạn text
        """
        raw = self.classifier(text)[0]

        sentiment = LABEL_MAP.get(raw["label"], "neutral")
        confidence = float(raw["score"])
        score = self.normalize_score(sentiment, confidence)

        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": round(confidence, 3),
        }

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """
        Predict sentiment cho nhiều đoạn text
        """
        outputs = self.classifier(texts)
        results = []

        for raw in outputs:
            sentiment = LABEL_MAP.get(raw["label"], "neutral")
            confidence = float(raw["score"])
            score = self.normalize_score(sentiment, confidence)

            results.append({
                "sentiment": sentiment,
                "score": score,
                "confidence": round(confidence, 3),
            })

        return results


# =====================
# SINGLETON INSTANCE
# =====================

_predictor: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """
    Load model đúng 1 lần duy nhất
    """
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


# =====================
# CONVENIENCE FUNCTION
# =====================

def predict(text: str) -> dict:
    """
    Hàm gọi nhanh (dùng trong route)
    """
    return get_predictor().predict(text)


# =====================
# LOCAL TEST
# =====================

if __name__ == "__main__":
    test_texts = [
        "Phòng sạch sẽ, giá hợp lý, nhân viên thân thiện.",
        "The room was dirty and the price was too high.",
        "Vị trí thuận tiện, tiện nghi tạm được.",
        "Dịch vụ quá tệ, không bao giờ quay lại.",
    ]

    predictor = SentimentPredictor()

    print("🔮 Testing predictions:\n")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
