"""Script push model lên Hugging Face Hub"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("./final_model")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./final_model")

print("Pushing model to hub...")
model.push_to_hub("qtrong/sentiment-analysis-model")

print("Pushing tokenizer to hub...")
tokenizer.push_to_hub("qtrong/sentiment-analysis-model")

print("✅ Done! Check: https://huggingface.co/qtrong/sentiment-analysis-model")
