"""
Module load model và predict sentiment
"""

from transformers import pipeline
from typing import Optional

# =====================
# CONFIG
# =====================

MODEL_PATH = "../final_model"

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


# =====================
# PREDICTOR CLASS
# =====================

class SentimentPredictor:
    """Class để load model và predict sentiment"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
        )
    
    @staticmethod
    def normalize_score(label: str, confidence: float) -> int:
        """Chuyển đổi label và confidence thành score 0-100"""
        if label == "positive":
            return int(50 + confidence * 50)
        if label == "negative":
            return int(50 - confidence * 50)
        return 50
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment cho một đoạn text
        
        Args:
            text: Đoạn văn bản cần phân tích
            
        Returns:
            dict với keys: sentiment, score, confidence
        """
        raw = self.classifier(text)[0]
        
        sentiment = LABEL_MAP[raw["label"]]
        confidence = raw["score"]
        score = self.normalize_score(sentiment, confidence)
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": round(confidence, 3),
        }
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict sentiment cho nhiều đoạn text"""
        return [self.predict(text) for text in texts]


# =====================
# CONVENIENCE FUNCTIONS
# =====================

_predictor: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """Get singleton predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


def predict(text: str) -> dict:
    """Quick predict function"""
    return get_predictor().predict(text)


# =====================
# MAIN
# =====================

if __name__ == "__main__":
    # Test predict
    test_texts = [
        "Phòng sạch sẽ, giá hợp lý, nhân viên thân thiện.",
        "The room was dirty and the price was too high.",
        "Vị trí thuận tiện, tiện nghi tạm được.",
    ]
    
    predictor = SentimentPredictor()
    
    print("🔮 Testing predictions:\n")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
