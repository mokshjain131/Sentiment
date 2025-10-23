# Original Model Training with Class Weights
# This is the baseline model but with class weight balancing

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# Setup
LABELS = ["negative", "neutral", "positive"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# Convert labels to IDs
df['label_id'] = df['label'].map(label2id)

# ============================================================
# CALCULATE CLASS WEIGHTS (to handle imbalance)
# ============================================================
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Calculate inverse frequency weights
class_counts = df['label'].value_counts().sort_index()
print(f"\n📊 Class distribution:")
for label, count in class_counts.items():
    print(f"   {label}: {count} ({100*count/len(df):.1f}%)")

# Compute class weights (higher weight for minority class)
class_weights_sklearn = compute_class_weight(
    'balanced',
    classes=np.array(LABELS),
    y=df['label'].values
)

class_weights = torch.tensor(class_weights_sklearn, dtype=torch.float32)
print(f"\n⚖️ Class weights (to balance training):")
for label, weight in zip(LABELS, class_weights):
    print(f"   {label}: {weight:.3f}")

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"\n📊 Train: {len(train_df)}, Validation: {len(val_df)}")

# Create PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load model and tokenizer
print("📦 Loading FinBERT model...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# Create datasets
train_dataset = SentimentDataset(
    train_df['text'].values,
    train_df['label_id'].values,
    tokenizer
)

val_dataset = SentimentDataset(
    val_df['text'].values,
    val_df['label_id'].values,
    tokenizer
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Larger batch on GPU
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to='none'  # Disable wandb
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    
    # Calculate per-class F1
    from sklearn.metrics import classification_report
    report = classification_report(labels, predictions, target_names=LABELS, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_negative': report['negative']['f1-score'],
        'f1_neutral': report['neutral']['f1-score'],
        'f1_positive': report['positive']['f1-score'],
        'f1_macro': report['macro avg']['f1-score']
    }

# ============================================================
# CUSTOM TRAINER WITH CLASS WEIGHTS
# ============================================================
from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    """Custom Trainer that uses class weights in loss calculation"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted cross-entropy loss
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Create Trainer with class weights
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights  # Pass the weights!
)

# Train!
print("\n🚀 Starting training with class weights...")
print("⏱️ Expected time: 3-5 minutes on T4 GPU\n")

trainer.train()

print("\n✅ Training complete!")

# Show training history
import pandas as pd
history = pd.DataFrame(trainer.state.log_history)
print("\n📊 Training History:")
print(history[['epoch', 'loss', 'eval_loss', 'eval_accuracy']].dropna())

# ============================================================
# SAVE MODEL TO GOOGLE DRIVE
# ============================================================
print("\n💾 Saving model to Google Drive...")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# Create directory
import os
output_dir = "/content/drive/MyDrive/sentiment_models/finetuned_original_weighted"
os.makedirs(output_dir, exist_ok=True)

# Save model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Model saved to Google Drive: {output_dir}")
print("\n📥 Download to your computer:")
print("   1. Go to Google Drive")
print("   2. Navigate to: MyDrive/sentiment_models/finetuned_original_weighted/")
print("   3. Download the entire folder")
print("   4. Place in your project: models/finetuned_original_weighted/")
