# AGGRESSIVE ANTI-OVERFITTING VERSION
# Use this if the above still overfits

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
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
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

print(f"📊 Train: {len(train_df)}, Validation: {len(val_df)}")

# Dataset class
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
# MAXIMUM DROPOUT + LAYER FREEZING
# ============================================================
print("📦 Loading FinBERT with AGGRESSIVE anti-overfitting...")
model_name = "ProsusAI/finbert"

# Very high dropout
config = AutoConfig.from_pretrained(model_name)
config.hidden_dropout_prob = 0.3  # Very high dropout
config.attention_probs_dropout_prob = 0.3
config.classifier_dropout = 0.3  # Classifier-specific dropout
config.num_labels = len(LABELS)  # Set in config
config.id2label = id2label
config.label2id = label2id

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

# FREEZE EARLY LAYERS (only fine-tune last few layers)
print("🔒 Freezing first 8 layers...")
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for i in range(8):  # Freeze first 8 of 12 layers
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

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

# ============================================================
# CONSERVATIVE TRAINING SETTINGS
# ============================================================
training_args = TrainingArguments(
    output_dir='./results_aggressive',  # Different directory
    
    # Shorter training
    num_train_epochs=3,
    
    # Smaller batch size (more updates, better generalization)
    per_device_train_batch_size=8,  # Reduced from 16
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch = 8*2 = 16
    
    # Lower learning rate
    learning_rate=1e-5,  # Half of original (2e-5)
    warmup_ratio=0.1,  # Warmup for 10% of training
    lr_scheduler_type='cosine',
    
    # Strong regularization
    weight_decay=0.1,  # Very high weight decay
    
    # Gradient clipping (prevent exploding gradients)
    max_grad_norm=0.5,  # Aggressive clipping
    
    # Evaluation
    eval_strategy='steps',
    eval_steps=50,  # Evaluate every 50 steps
    save_strategy='steps',
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_total_limit=1,  # Only keep best checkpoint
    
    # Logging
    logging_steps=25,
    
    # Misc
    report_to='none',
    seed=42,
    fp16=True,  # Mixed precision (faster + acts as regularization)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    
    from sklearn.metrics import f1_score
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
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

# Early stopping with patience=1 (very aggressive)
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=1,
    early_stopping_threshold=0.001
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
    class_weights=class_weights  # Pass the weights!
)

print("\n🚀 Starting CONSERVATIVE training (strong anti-overfitting)...")
print("⏱️ Expected time: 2-3 minutes\n")

trainer.train()

print("\n✅ Training complete!")

# Show results
import pandas as pd
history = pd.DataFrame(trainer.state.log_history)
print("\n📊 Training History:")
print(history[['epoch', 'loss', 'eval_loss', 'eval_accuracy', 'eval_f1_macro']].dropna())

# ============================================================
# SAVE MODEL with unique name
# ============================================================
print("\n💾 Saving model...")

# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

output_dir = "/content/drive/MyDrive/sentiment_models/finetuned_aggressive"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")
print("\n📥 Local path for your project: models/finetuned_aggressive/")
