"""
Test SOTA Models (State of the Art)
Compares performance of FinBERT against modern architectures (RoBERTa, DeBERTa)

Usage:
    python test_sota_models.py

Output:
    sota_comparison.csv - Predictions from all models
    reports/sota_models_analysis.txt - Detailed analysis report
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import torch
from pathlib import Path
import random
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load test data from FinancialPhraseBank dataset
def load_test_data(num_samples=200):
    """Load balanced test samples from the dataset"""
    # Try to find the dataset file
    dataset_path = Path('data/datasets/financialphrasebank.csv')
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return []

    df = pd.read_csv(dataset_path)
    
    # Get balanced samples from each class
    samples_per_class = num_samples // 3
    
    test_samples = []
    for label in ['positive', 'negative', 'neutral']:
        class_data = df[df['label'] == label]
        if len(class_data) < samples_per_class:
            sampled = class_data.sample(n=len(class_data), random_state=42)
        else:
            sampled = class_data.sample(n=samples_per_class, random_state=42)
        for _, row in sampled.iterrows():
            test_samples.append((row['text'], row['label']))
    
    # Shuffle to mix labels
    random.seed(42)
    random.shuffle(test_samples)
    
    print(f"✅ Loaded {len(test_samples)} test sentences:")
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for _, lbl in test_samples if lbl == label)
        print(f"   {label}: {count} samples")
    print()
    return test_samples

# Load test sentences
print("Loading test data from FinancialPhraseBank dataset...")
TEST_SENTENCES = load_test_data(num_samples=200)

def load_model_pipeline(model_name, description):
    """Generic function to load a model pipeline"""
    print(f"📦 Loading {description}...")
    print(f"   Model: {model_name}")
    try:
        pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
        print(f"   ✅ Loaded successfully")
        return pipe
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return None

def map_label(label, model_type):
    """Map model-specific labels to standard positive/negative/neutral"""
    label = label.lower()
    
    # FinBERT & General
    if label in ['positive', 'negative', 'neutral']:
        return label
    
    # Label_0/1/2 mapping (common in RoBERTa/DeBERTa)
    # Usually: 0=Negative, 1=Neutral, 2=Positive (Check specific model card if unsure)
    if label == 'label_0': return 'negative'
    if label == 'label_1': return 'neutral'
    if label == 'label_2': return 'positive'
    
    # Twitter RoBERTa
    if label == 'joy': return 'positive'
    if label == 'sadness': return 'negative'
    
    return label

def predict(text, pipeline, model_type):
    """Predict using a pipeline"""
    if pipeline is None:
        return "N/A", 0.0
    
    try:
        # Truncate to 512 tokens to avoid errors
        result = pipeline(text, truncation=True, max_length=512)[0]
        raw_label = result['label']
        confidence = result['score']
        
        mapped_label = map_label(raw_label, model_type)
        return mapped_label, confidence
    except Exception as e:
        return "ERROR", 0.0

def main():
    if not TEST_SENTENCES:
        return

    print("=" * 70)
    print("🧪 TESTING SOTA MODELS (State of the Art Comparison)")
    print("=" * 70)
    print()
    
    # 1. FinBERT (Baseline)
    finbert = load_model_pipeline("ProsusAI/finbert", "FinBERT (Baseline)")
    
    # 2. Financial RoBERTa (DistilRoBERTa)
    # mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
    # Labels: 0: negative, 1: neutral, 2: positive
    fin_roberta = load_model_pipeline("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "Financial DistilRoBERTa")
    
    # 3. Twitter RoBERTa (General SOTA)
    # cardiffnlp/twitter-roberta-base-sentiment-latest
    # Labels: negative, neutral, positive
    gen_roberta = load_model_pipeline("cardiffnlp/twitter-roberta-base-sentiment-latest", "Twitter RoBERTa (General SOTA)")

    print()
    print("=" * 70)
    print("🚀 STARTING PREDICTIONS")
    print("=" * 70)
    print()
    
    results = []
    
    for i, (sentence, true_label) in enumerate(TEST_SENTENCES, 1):
        # FinBERT
        f_pred, f_conf = predict(sentence, finbert, "finbert")
        
        # Financial RoBERTa
        fr_pred, fr_conf = predict(sentence, fin_roberta, "fin_roberta")
        
        # General RoBERTa
        gr_pred, gr_conf = predict(sentence, gen_roberta, "gen_roberta")
        
        results.append({
            'sentence': sentence,
            'true_label': true_label,
            'finbert_pred': f_pred,
            'finbert_correct': f_pred == true_label,
            'fin_roberta_pred': fr_pred,
            'fin_roberta_correct': fr_pred == true_label,
            'gen_roberta_pred': gr_pred,
            'gen_roberta_correct': gr_pred == true_label
        })
        
        if i % 20 == 0:
            print(f"Processed {i}/{len(TEST_SENTENCES)} sentences...")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Accuracies
    f_acc = df['finbert_correct'].mean() * 100
    fr_acc = df['fin_roberta_correct'].mean() * 100
    gr_acc = df['gen_roberta_correct'].mean() * 100
    
    print()
    print("=" * 70)
    print("🏆 RESULTS")
    print("=" * 70)
    print(f"FinBERT (Baseline):        {f_acc:.1f}%")
    print(f"Financial DistilRoBERTa:   {fr_acc:.1f}%")
    print(f"Twitter RoBERTa (General): {gr_acc:.1f}%")
    print()
    
    # Save results
    df.to_csv("sota_comparison.csv", index=False)
    print("Saved detailed results to sota_comparison.csv")

if __name__ == "__main__":
    main()
