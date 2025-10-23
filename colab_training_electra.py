"""
Google Colab Training Script: ELECTRA Model
Sample-efficient, discriminator-based architecture (often outperforms BERT)

Features:
- 110M parameters (same as BERT)
- More efficient training (generator-discriminator)
- Often achieves better accuracy than BERT
- Same inference speed as BERT

Usage in Colab:
1. Runtime → Change runtime type → T4 GPU
2. Run this entire cell
3. Model saves to Google Drive
"""

# ============================================================
# INSTALL DEPENDENCIES
# ============================================================
!pip install transformers datasets torch pandas scikit-learn pyarrow -q

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

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "google/electra-base-discriminator"
MODEL_OUTPUT_NAME = "electra_financial"

LABELS = ["negative", "neutral", "positive"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# Training configuration - ANTI-OVERFITTING OPTIMIZED
TRAINING_CONFIG = {
    'num_train_epochs': 3,  # Allow 3 epochs but use early stopping
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 32,
    'learning_rate': 2e-5,  # Standard for ELECTRA
    'weight_decay': 0.1,  # Increased from 0.05 for stronger regularization
    'warmup_steps': 150,  # Increased warmup for smoother training
    'warmup_ratio': 0.1,  # 10% of training for warmup
    'logging_steps': 50,
    'eval_strategy': 'steps',  # Evaluate more frequently
    'eval_steps': 100,  # Evaluate every 100 steps
    'save_strategy': 'steps',
    'save_steps': 100,
    'save_total_limit': 2,  # Keep only best 2 checkpoints
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'report_to': 'none',
    'fp16': True,  # Mixed precision for speed
    'gradient_accumulation_steps': 2,  # Effective batch size = 32
    'max_grad_norm': 1.0,
    'lr_scheduler_type': 'cosine',  # Cosine decay with restarts
    'dataloader_drop_last': True,  # Drop incomplete batches
    'label_smoothing_factor': 0.1,  # Label smoothing to prevent overconfidence
}

# Test sentences
TEST_SENTENCES = [
    "Company profits surged 50% exceeding all expectations",
    "The firm reported quarterly earnings as expected",
    "Stock plunged amid scandal and massive layoffs",
    "The company is not unprofitable",
    "Analysts raise price target following strong guidance",
    "Disappointing revenue misses estimates significantly",
]

print("="*70)
print("🚀 TRAINING: ELECTRA for Financial Sentiment Analysis")
print("="*70)
print(f"\nModel: {MODEL_NAME}")
print(f"Expected params: ~110M (same as BERT)")
print(f"Expected speed: Similar to BERT")
print(f"Expected accuracy: 82-84% (often outperforms BERT)\n")

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ============================================================
# LOAD DATASET
# ============================================================
print("📥 Downloading FinancialPhraseBank dataset...")
dataset = load_dataset("mltrev23/financial-sentiment-analysis")
df = dataset['train'].to_pandas()
df = df.rename(columns={'Sentence': 'text', 'Sentiment': 'label'})
df['label'] = df['label'].str.lower()

print(f"✅ Dataset loaded: {len(df)} samples")
print(f"\n📊 Label distribution:")
print(df['label'].value_counts())
print(f"\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100)

# Convert labels to IDs
df['label_id'] = df['label'].map(label2id)

# ============================================================
# TRAIN/VAL SPLIT
# ============================================================
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

print(f"\n📊 Split: Train={len(train_df)}, Validation={len(val_df)}")

# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================
class_weights = compute_class_weight(
    'balanced',
    classes=np.array(LABELS),
    y=train_df['label'].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print(f"\n⚖️ Class weights (to handle imbalance):")
for label, weight in zip(LABELS, class_weights):
    print(f"   {label}: {weight:.3f}")

# ============================================================
# PYTORCH DATASET
# ============================================================
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

# ============================================================
# WEIGHTED TRAINER (for class imbalance)
# ============================================================
class WeightedTrainer(Trainer):
    """Custom Trainer with class weights"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ============================================================
# METRICS
# ============================================================
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_per_class = f1_score(labels, predictions, average=None)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_negative': f1_per_class[0],
        'f1_neutral': f1_per_class[1],
        'f1_positive': f1_per_class[2],
        'precision': precision,
        'recall': recall
    }

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
print(f"\n📦 Loading ELECTRA tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Configure model
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = len(LABELS)
config.id2label = id2label
config.label2id = label2id

# Add dropout for regularization - Increased to combat overfitting
if hasattr(config, 'hidden_dropout_prob'):
    config.hidden_dropout_prob = 0.3  # Increased from 0.2
if hasattr(config, 'attention_probs_dropout_prob'):
    config.attention_probs_dropout_prob = 0.3  # Increased from 0.2
if hasattr(config, 'classifier_dropout'):
    config.classifier_dropout = 0.3  # Increased from 0.2

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True
)

# Model size
model_size = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅ Model loaded: {model_size:.1f}M parameters")

# ============================================================
# CREATE DATASETS
# ============================================================
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

# ============================================================
# TRAINING ARGUMENTS
# ============================================================
output_dir = './results_electra'
training_args = TrainingArguments(
    output_dir=output_dir,
    **TRAINING_CONFIG
)

# ============================================================
# CREATE TRAINER
# ============================================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
            early_stopping_threshold=0.001  # Minimum improvement required
        )
    ]
)

# ============================================================
# TRAIN MODEL
# ============================================================
print(f"\n{'='*70}")
print("🏋️ TRAINING STARTED")
print(f"{'='*70}\n")

start_time = time.time()
train_result = trainer.train()
training_time = time.time() - start_time

print(f"\n✅ Training complete! Time: {training_time/60:.2f} minutes")

# ============================================================
# TRAINING HISTORY
# ============================================================
history = pd.DataFrame(trainer.state.log_history)
print(f"\n📈 Training History:")
print(history[['epoch', 'loss', 'eval_loss', 'eval_accuracy', 'eval_f1_macro']].dropna())

# ============================================================
# FINAL EVALUATION
# ============================================================
print(f"\n{'='*70}")
print("📊 FINAL EVALUATION")
print(f"{'='*70}\n")

eval_results = trainer.evaluate()

print("Validation Metrics:")
print(f"  Loss: {eval_results['eval_loss']:.4f}")
print(f"  Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
print(f"  F1 Macro: {eval_results['eval_f1_macro']:.4f}")
print(f"  Precision: {eval_results['eval_precision']:.4f}")
print(f"  Recall: {eval_results['eval_recall']:.4f}")
print(f"\nPer-Class F1 Scores:")
print(f"  Negative: {eval_results['eval_f1_negative']:.4f}")
print(f"  Neutral:  {eval_results['eval_f1_neutral']:.4f}")
print(f"  Positive: {eval_results['eval_f1_positive']:.4f}")

# ============================================================
# TEST PREDICTIONS
# ============================================================
print(f"\n{'='*70}")
print("🧪 TEST PREDICTIONS")
print(f"{'='*70}\n")

model.eval()
for sentence in TEST_SENTENCES:
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = probs.argmax().item()
        confidence = probs[0][pred_label].item()
    
    predicted = id2label[pred_label]
    print(f"Text: '{sentence}'")
    print(f"→ {predicted.upper()} ({confidence*100:.2f}% confidence)")
    print(f"  Probabilities: neg={probs[0][0]*100:.1f}%, neu={probs[0][1]*100:.1f}%, pos={probs[0][2]*100:.1f}%\n")

# ============================================================
# MEASURE INFERENCE SPEED
# ============================================================
print(f"{'='*70}")
print("⚡ MEASURING INFERENCE SPEED")
print(f"{'='*70}\n")

test_texts = val_df['text'].head(100).tolist()

start = time.time()
for text in test_texts:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model(**inputs)

inference_time = time.time() - start
samples_per_sec = len(test_texts) / inference_time

print(f"Processed {len(test_texts)} samples in {inference_time:.2f} seconds")
print(f"Speed: {samples_per_sec:.2f} samples/second")
print(f"Expected: Similar to BERT-base\n")

# ============================================================
# SAVE MODEL
# ============================================================
save_dir = f"/content/drive/MyDrive/sentiment_models/{MODEL_OUTPUT_NAME}"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"{'='*70}")
print("💾 MODEL SAVED")
print(f"{'='*70}\n")
print(f"Location: {save_dir}")
print(f"\nTo use this model locally:")
print(f"1. Download folder from Google Drive")
print(f"2. Place in: models/Colab/{MODEL_OUTPUT_NAME}/")
print(f"3. Load with: AutoModelForSequenceClassification.from_pretrained('models/Colab/{MODEL_OUTPUT_NAME}/')")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE - ELECTRA")
print(f"{'='*70}\n")

print("📊 Summary:")
print(f"  Model: ELECTRA ({model_size:.1f}M parameters)")
print(f"  Training Time: {training_time/60:.2f} minutes")
print(f"  Validation Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
print(f"  F1 Macro: {eval_results['eval_f1_macro']:.4f}")
print(f"  F1 Negative: {eval_results['eval_f1_negative']:.4f}")
print(f"  Inference Speed: {samples_per_sec:.2f} samples/second")
print(f"  Model Size: ~440 MB (same as BERT)")

print(f"\n💡 ELECTRA Advantages:")
print(f"  ✅ Generator-discriminator architecture")
print(f"  ✅ More sample-efficient training")
print(f"  ✅ Often outperforms BERT (1-2% higher accuracy)")
print(f"  ✅ Same model size as BERT")
print(f"  ✅ Similar inference speed")
print(f"  ✅ Better for smaller datasets")

print(f"\n🎯 Best Use Cases:")
print(f"  • When accuracy is the top priority")
print(f"  • Limited training data scenarios")
print(f"  • When you want cutting-edge architecture")
print(f"  • Balanced performance requirements")
print(f"  • Research and experimentation")

print(f"\n📊 Expected Performance vs Competitors:")
print(f"  • Accuracy: ~1-2% better than BERT")
print(f"  • Accuracy: ~2-3% better than DistilBERT")
print(f"  • Speed: Similar to BERT")
print(f"  • Speed: ~40% slower than DistilBERT")
print(f"  • Size: Same as BERT, larger than DistilBERT")

print(f"\n{'='*70}\n")
