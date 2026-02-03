"""
Google Colab Training Script: DeBERTa v3 Model
State-of-the-Art (SOTA) performance for text classification.

Features:
- DeBERTa (Decoding-enhanced BERT with disentangled attention)
- Generally outperforms BERT, RoBERTa, and XLNet
- Microsoft's v3 improves pre-training efficiency
- Best choice for maximizing accuracy

Usage in Colab:
1. Runtime → Change runtime type → T4 GPU
2. Run this entire cell
3. Model saves to Google Drive
"""

# ============================================================
# INSTALL DEPENDENCIES
# ============================================================
!pip install transformers datasets torch pandas scikit-learn pyarrow sentencepiece accelerate -q

# ============================================================
# IMPORTS
# ============================================================
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_OUTPUT_NAME = "deberta_financial"

# Hyperparameters (DeBERTa often benefits from smaller LR)
BATCH_SIZE = 16  # DeBERTa is larger, might need smaller batch size
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 128

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

# Load FinancialPhraseBank (using 50% agreement split for cleaner data)
print("📥 Loading FinancialPhraseBank dataset...")
dataset = load_dataset("financial_phrasebank", "sentences_50agree")

# Convert to pandas for easier handling
df = pd.DataFrame(dataset['train'])
print(f"✅ Loaded {len(df)} sentences")

# Map labels (0: negative, 1: neutral, 2: positive)
# Note: FinancialPhraseBank labels are 0:negative, 1:neutral, 2:positive
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Split data (80% train, 10% val, 10% test)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

# ============================================================
# TOKENIZATION
# ============================================================

print(f"⚙️ Loading Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    
    class FinancialDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.tolist()

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
            
    return FinancialDataset(encodings, labels)

train_dataset = tokenize_data(train_texts, train_labels)
val_dataset = tokenize_data(val_texts, val_labels)
test_dataset = tokenize_data(test_texts, test_labels)

# ============================================================
# MODEL SETUP
# ============================================================

print(f"🏗️ Initializing Model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=3,
    id2label=label_map,
    label2id={v: k for k, v in label_map.items()}
)

# ============================================================
# TRAINING
# ============================================================

training_args = TrainingArguments(
    output_dir=f'./results/{MODEL_OUTPUT_NAME}',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=LEARNING_RATE,
    fp16=True, # Use mixed precision for speed
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': (predictions == labels).mean()
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🚀 Starting Training...")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"✅ Training complete in {(end_time - start_time)/60:.2f} minutes")

# ============================================================
# EVALUATION
# ============================================================

print("\n📊 Evaluating on Test Set...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

print("\nClassification Report:")
print(classification_report(test_labels, preds, target_names=["negative", "neutral", "positive"]))

# ============================================================
# SAVING
# ============================================================

save_path = f"./drive/MyDrive/models/{MODEL_OUTPUT_NAME}"
print(f"\n💾 Saving model to {save_path}...")
# Create directory if running locally or ensure drive is mounted
# trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)
print("Done! (Uncomment save lines in Colab)")
