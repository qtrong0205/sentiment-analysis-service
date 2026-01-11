# 🏨 Sentiment Analysis - Hotel Reviews

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dự án phân tích cảm xúc (Sentiment Analysis) cho review khách sạn, hỗ trợ **tiếng Việt** và **tiếng Anh**.

## ✨ Tính năng

- 🔍 Phân tích cảm xúc: Negative / Neutral / Positive
- 🌐 Hỗ trợ đa ngôn ngữ: Tiếng Việt & Tiếng Anh
- 🚀 REST API với FastAPI
- 📊 Điểm sentiment từ 0-100
- ⚡ Inference nhanh với Transformers

## 📁 Cấu trúc dự án

```
sentiment-analysis/
├── data/
│   └── sample.csv          # Dữ liệu mẫu
├── training/
│   └── train.py            # Script huấn luyện model
├── inference/
│   └── predict.py          # Load model & predict
├── server/
│   └── app.py              # FastAPI server
├── final_model/            # Model đã train (không có trong repo)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```


## 📊 Huấn luyện model

```bash
cd training
python train.py
```

**Cấu hình training** có thể chỉnh sửa trong `training/train.py`:
- `MODEL_NAME`: Base model (mặc định: `vinai/phobert-base`)
- `DATA_PATH`: Đường dẫn dữ liệu
- `NUM_TRAIN_EPOCHS`: Số epoch

## 🔮 Sử dụng

### Từ Python

```python
from inference.predict import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("Phòng sạch sẽ, nhân viên thân thiện.")
print(result)
# {'sentiment': 'positive', 'score': 95, 'confidence': 0.98}
```

### Chạy test

```bash
cd inference
python predict.py
```

## 🌐 Chạy API Server

### Cách 1: Uvicorn (development)

```bash
# Từ thư mục gốc
python -m uvicorn server.app:app --reload --port 8000

# Hoặc từ thư mục server
cd server
uvicorn app:app --reload --port 8000
```

### Cách 2: Chạy trực tiếp

```bash
cd server
python app.py
```

Server sẽ chạy tại: http://localhost:8000

### 📚 API Documentation

Truy cập Swagger UI: http://localhost:8000/docs

### API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict một text |
| `POST` | `/predict/batch` | Predict nhiều text |

### Ví dụ Request

**Single prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Phòng sạch sẽ, giá hợp lý."}'
```

**Response:**

```json
{
  "sentiment": "positive",
  "score": 92,
  "confidence": 0.95
}
```

**Batch prediction:**

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Phòng đẹp", "Giá quá cao", "Tạm được"]}'
```

## 📝 Labels & Scoring

| Label | Ý nghĩa | Score Range |
|-------|---------|-------------|
| `negative` | Tiêu cực | 0 - 49 |
| `neutral` | Trung lập | 50 |
| `positive` | Tích cực | 51 - 100 |

## 🛠 Tech Stack

- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **ML Library**: [Transformers](https://huggingface.co/transformers/) (Hugging Face)
- **Model**: XLM-RoBERTa (multilingual)
- **Server**: Uvicorn

## 📦 Requirements

```
fastapi
uvicorn
transformers
torch
datasets
scikit-learn
pandas
protobuf
sentencepiece
```

## 🤝 Contributing

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Author

**Your Name**
- GitHub: [qtrong0205](https://github.com/qtrong0205)
- Email: quoctrong02052006@gmail.com

---

⭐ Nếu project hữu ích, hãy cho một star nhé!
