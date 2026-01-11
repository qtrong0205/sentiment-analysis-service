"""
Script huấn luyện model phân loại sentiment
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# =====================
# CONFIG
# =====================

MODEL_NAME = "vinai/phobert-base"  # hoặc "bert-base-multilingual-cased"
DATA_PATH = "../data/sample.csv"
OUTPUT_DIR = "../final_model"
NUM_LABELS = 3  # negative=0, neutral=1, positive=2

TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
}


# =====================
# LOAD DATA
# =====================

def load_data(path: str):
    """Load CSV data và chia train/val"""
    df = pd.read_csv(path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df


# =====================
# TOKENIZE
# =====================

def tokenize_data(df: pd.DataFrame, tokenizer):
    """Tokenize text data"""
    dataset = Dataset.from_pandas(df)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    
    return dataset.map(tokenize_fn, batched=True)


# =====================
# TRAIN
# =====================

def train():
    print("📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    print("📂 Loading data...")
    train_df, val_df = load_data(DATA_PATH)
    
    print("🔤 Tokenizing...")
    train_dataset = tokenize_data(train_df, tokenizer)
    val_dataset = tokenize_data(val_df, tokenizer)

    print("🚀 Training...")
    training_args = TrainingArguments(**TRAINING_ARGS)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    print("💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
