"""
Test Pre-trained Models (Before Fine-tuning)
Compares baseline performance of FinBERT, DistilBERT, and ELECTRA

Usage:
    python test_pretrained_models.py

Output:
    pretrained_comparison.csv - Predictions from all 3 models
    reports/pretrained_models_analysis.txt - Detailed analysis report
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
def load_test_data(num_samples=50):
    """Load balanced test samples from the dataset"""
    df = pd.read_csv('data/datasets/financialphrasebank.csv')
    
    # Get balanced samples from each class
    samples_per_class = num_samples // 3
    
    test_samples = []
    for label in ['positive', 'negative', 'neutral']:
        class_data = df[df['label'] == label]
        if len(class_data) < samples_per_class:
            print(f"  Warning: Only {len(class_data)} samples available for {label}, sampling {len(class_data)}")
            sampled = class_data.sample(n=len(class_data), random_state=42)
        else:
            sampled = class_data.sample(n=samples_per_class, random_state=42)
        for _, row in sampled.iterrows():
            test_samples.append((row['text'], row['label']))
    
    # Shuffle to mix labels
    random.seed(42)
    random.shuffle(test_samples)
    
    print(f" Loaded {len(test_samples)} test sentences:")
    for label in ['positive', 'negative', 'neutral']:
        count = sum(1 for _, lbl in test_samples if lbl == label)
        print(f"   {label}: {count} samples")
    print()
    
    return test_samples

# Load test sentences from dataset
print("Loading test data from FinancialPhraseBank dataset...")
TEST_SENTENCES = load_test_data(num_samples=200)
print(f" Loaded {len(TEST_SENTENCES)} test sentences (balanced across positive/negative/neutral)")
print()

def load_finbert():
    """Load FinBERT (already fine-tuned for financial sentiment)"""
    print("¦ Loading FinBERT (financial sentiment specialist)...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # FinBERT has specific label mapping
    # Check the model config for actual labels
    if hasattr(model.config, 'id2label'):
        label_map = model.config.id2label
    else:
        # FinBERT standard mapping
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
    
    print(f"    FinBERT loaded (labels: {list(label_map.values())})")
    return tokenizer, model, label_map

def load_distilbert():
    """Load DistilBERT (fine-tuned on general sentiment SST-2)"""
    print("¦ Loading DistilBERT (general sentiment)...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Use pipeline for simplicity (SST-2 has POSITIVE/NEGATIVE only)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name
    )
    
    print("    DistilBERT loaded (labels: POSITIVE, NEGATIVE)")
    print("     Note: SST-2 has NO NEUTRAL label!")
    return sentiment_pipeline

def load_electra_sentiment():
    """Load ELECTRA fine-tuned for emotion/sentiment"""
    print("¦ Loading ELECTRA (emotion detection)...")
    
    # ELECTRA doesn't have a standard sentiment model
    # Use emotion-finetuned version as best alternative
    try:
        model_name = "bhadresh-savani/electra-base-emotion"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
        print("    ELECTRA loaded (emotion labels)")
        print("     Note: Trained on emotions, not financial sentiment")
        return sentiment_pipeline
    except Exception as e:
        print(f"     Could not load ELECTRA emotion model: {e}")
        print("     ELECTRA predictions will be marked as N/A")
        return None

def predict_finbert(text, tokenizer, model, label_map):
    """Predict with FinBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax().item()
        confidence = probs[0][pred_idx].item()
        all_probs = probs[0].tolist()
    
    return label_map[pred_idx], confidence, all_probs

def predict_distilbert(text, pipeline):
    """Predict with DistilBERT (SST-2 only has positive/negative)"""
    result = pipeline(text, truncation=True, max_length=512)[0]
    label = result['label'].lower()
    confidence = result['score']
    
    # SST-2 doesn't have neutral, just positive/negative
    return label, confidence, None

def predict_electra(text, pipeline):
    """Predict with ELECTRA"""
    if pipeline is None:
        return "N/A", 0.0, None
    
    try:
        result = pipeline(text, truncation=True, max_length=512)[0]
        label = result['label'].lower()
        confidence = result['score']
        # Map ELECTRA emotion labels to sentiment
        if label == 'joy':
            mapped_label = 'positive'
        elif label == 'sadness':
            mapped_label = 'negative'
        elif label in ['anger', 'fear', 'surprise', 'disgust']:  # treat as neutral for this task
            mapped_label = 'neutral'
        else:
            mapped_label = label  # fallback
        return mapped_label, confidence, label  # also return original label for analysis
    except Exception as e:
        return "ERROR", 0.0, None

def main():
    print("=" * 70)
    print("§ª TESTING PRE-TRAINED MODELS (Before Fine-tuning)")
    print("=" * 70)
    print()
    print("This script tests 3 models in their PRE-TRAINED state:")
    print("1. FinBERT - Already trained on financial sentiment")
    print("2. DistilBERT - Trained on general sentiment (movie reviews)")
    print("3. ELECTRA - Trained on emotion detection")
    print()
    print(f"Testing on {len(TEST_SENTENCES)} financial sentences...")
    print()
    print("=" * 70)
    print()
    
    # Load models
    print("¥ Loading models from Hugging Face...\n")
    finbert_tokenizer, finbert_model, finbert_labels = load_finbert()
    distilbert_pipeline = load_distilbert()
    electra_pipeline = load_electra_sentiment()
    
    print()
    print(" All models loaded successfully!")
    print()
    print("=" * 70)
    print()
    
    # Test all sentences
    results = []
    
    for i, (sentence, true_label) in enumerate(TEST_SENTENCES, 1):
        try:
            print(f"[{i}/{len(TEST_SENTENCES)}] Testing: '{sentence[:55]}...'")
            print(f"   True Label: {true_label.upper()}")
            # FinBERT prediction
            finbert_label, finbert_conf, finbert_probs = predict_finbert(
                sentence, finbert_tokenizer, finbert_model, finbert_labels
            )
            # DistilBERT prediction
            distilbert_label, distilbert_conf, _ = predict_distilbert(
                sentence, distilbert_pipeline
            )
            # ELECTRA prediction (returns mapped, conf, original)
            electra_label, electra_conf, electra_orig_label = predict_electra(
                sentence, electra_pipeline
            )
            # Store results
            result = {
                'sentence': sentence,
                'true_label': true_label,
                'finbert_prediction': finbert_label,
                'finbert_confidence': round(finbert_conf, 4),
                'finbert_correct': finbert_label == true_label,
                'distilbert_prediction': distilbert_label,
                'distilbert_confidence': round(distilbert_conf, 4),
                'distilbert_correct': distilbert_label == true_label,
                'electra_prediction': electra_label,
                'electra_confidence': round(electra_conf, 4),
                'electra_correct': electra_label == true_label,
                'electra_raw_label': electra_orig_label
            }
            # Add probability details for FinBERT (has 3 classes)
            if finbert_probs:
                result['finbert_prob_positive'] = round(finbert_probs[0], 4)
                result['finbert_prob_negative'] = round(finbert_probs[1], 4)
                result['finbert_prob_neutral'] = round(finbert_probs[2], 4)
            results.append(result)
            # Display predictions
            finbert_mark = "" if finbert_label == true_label else ""
            distilbert_mark = "" if distilbert_label == true_label else ""
            electra_mark = "" if electra_label == true_label else ""
            print(f"   FinBERT:    {finbert_label:8s} ({finbert_conf*100:.1f}%) {finbert_mark}")
            print(f"   DistilBERT: {distilbert_label:8s} ({distilbert_conf*100:.1f}%) {distilbert_mark}")
            print(f"   ELECTRA:    {electra_label:8s} ({electra_conf*100:.1f}%) {electra_mark} (raw: {electra_orig_label})")
            print()
        except Exception as e:
            print(f" Error processing sentence {i}: {e}")
            print("   Skipping this sentence and continuing...")
            print()
            continue
    
    # Create DataFrame (always, even if results is empty)
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print(" No results to analyze. Check for errors above.")
        return
    
    print(f" Processed {len(df)} sentences successfully (out of {len(TEST_SENTENCES)})")
    print()
    
    # --- METRICS: precision, recall, f1, accuracy ---
    y_true = df['true_label']
    y_pred_finbert = df['finbert_prediction']
    y_pred_distilbert = df['distilbert_prediction']
    y_pred_electra = df['electra_prediction']

    print("\nMETRICS (Precision, Recall, F1, Accuracy):\n")
    for model, y_pred in zip(['FinBERT', 'DistilBERT', 'ELECTRA'], [y_pred_finbert, y_pred_distilbert, y_pred_electra]):
        print(f"{model}:")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    
    # Save to CSV
    output_file = "pretrained_comparison.csv"
    df.to_csv(output_file, index=False)
    
    # ============================================
    # ACCURACY COMPARISON REPORT
    # ============================================
    
    print("=" * 70)
    print(" ACCURACY COMPARISON REPORT - PRE-TRAINED MODELS")
    print("=" * 70)
    print()
    
    # Overall accuracy
    finbert_acc = df['finbert_correct'].sum() / len(df) * 100
    distilbert_acc = df['distilbert_correct'].sum() / len(df) * 100
    electra_acc = df['electra_correct'].sum() / len(df) * 100
    
    print(" OVERALL ACCURACY:")
    print(f"   FinBERT:    {finbert_acc:5.1f}% ({df['finbert_correct'].sum()}/{len(df)} correct)")
    print(f"   DistilBERT: {distilbert_acc:5.1f}% ({df['distilbert_correct'].sum()}/{len(df)} correct)")
    print(f"   ELECTRA:    {electra_acc:5.1f}% ({df['electra_correct'].sum()}/{len(df)} correct)")
    print()
    
    # Per-class accuracy
    print(" PER-CLASS ACCURACY:")
    for label in ['positive', 'negative', 'neutral']:
        class_df = df[df['true_label'] == label]
        if len(class_df) > 0:
            finbert_class_acc = class_df['finbert_correct'].sum() / len(class_df) * 100
            distilbert_class_acc = class_df['distilbert_correct'].sum() / len(class_df) * 100
            electra_class_acc = class_df['electra_correct'].sum() / len(class_df) * 100
            
            print(f"   {label.upper()} ({len(class_df)} samples):")
            print(f"      FinBERT:    {finbert_class_acc:5.1f}%")
            print(f"      DistilBERT: {distilbert_class_acc:5.1f}%")
            print(f"      ELECTRA:    {electra_class_acc:5.1f}%")
    print()
    
    # Confusion matrix-style breakdown
    print(" PREDICTION BREAKDOWN (Confusion Matrix):")
    for model in ['finbert', 'distilbert', 'electra']:
        print(f"\n   {model.upper()}:")
        pred_col = f'{model}_prediction'
        for true_label in ['positive', 'negative', 'neutral']:
            class_preds = df[df['true_label'] == true_label][pred_col].value_counts()
            if len(class_preds) > 0:
                pred_str = ", ".join([f"{k}:{v}" for k, v in dict(class_preds).items()])
                print(f"      True {true_label:8s}  {pred_str}")
    print()
    
    # Winner announcement
    accuracies = {
        'FinBERT': finbert_acc,
        'DistilBERT': distilbert_acc,
        'ELECTRA': electra_acc
    }
    winner = max(accuracies, key=accuracies.get)
    
    print("=" * 70)
    print(f" WINNER (Pre-trained Baseline): {winner} with {accuracies[winner]:.1f}% accuracy")
    print("=" * 70)
    print()
    
    print(f" Full results with true labels saved to: {output_file}")
    print()
    
    print("=" * 70)
    print(" ANALYSIS & SUMMARY")
    print("=" * 70)
    print()
    
    # Basic statistics
    print(f"Total Sentences Tested: {len(df)}")
    print()
    
    # FinBERT distribution (most reliable)
    print("FinBERT Predictions Distribution:")
    finbert_counts = df['finbert_prediction'].value_counts()
    for label, count in finbert_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {label:8s}: {count:2d} ({pct:.1f}%)")
    print()
    
    # Agreement analysis (FinBERT vs DistilBERT on positive/negative only)
    finbert_pos_neg = df[df['finbert_prediction'].isin(['positive', 'negative'])]
    if len(finbert_pos_neg) > 0:
        agreement = 0
        for _, row in finbert_pos_neg.iterrows():
            if row['finbert_prediction'] == row['distilbert_prediction']:
                agreement += 1
        agreement_pct = (agreement / len(finbert_pos_neg)) * 100
        print(f"FinBERT vs DistilBERT Agreement (pos/neg only):")
        print(f"   {agreement}/{len(finbert_pos_neg)} sentences ({agreement_pct:.1f}%)")
        print()
    
    # Show sample results
    print("Sample Results (First 5 sentences):")
    display_cols = ['sentence', 'finbert_prediction', 'distilbert_prediction', 'electra_prediction']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df.head()[available_cols].to_string(index=False))
    print()
    
    print(f" Full results saved to: {output_file}")
    print()
    
    # Model notes
    print("=" * 70)
    print(" MODEL INTERPRETATIONS")
    print("=" * 70)
    print()
    print("1£  FinBERT (ProsusAI/finbert):")
    print("    Already fine-tuned on financial text")
    print("    Has 3 labels: positive, negative, neutral")
    print("    Best for financial sentiment OUT OF THE BOX")
    print("    Should show most accurate predictions")
    print("    This is your baseline to beat!")
    print()
    print("2£  DistilBERT (SST-2 model):")
    print("     Fine-tuned on movie reviews (SST-2)")
    print("     Only has 2 labels: POSITIVE, NEGATIVE")
    print("     NO NEUTRAL LABEL (forces binary choice)")
    print("     May not understand financial context")
    print("    After fine-tuning: will learn 3-class + finance")
    print()
    print("3£  ELECTRA (emotion model):")
    print("     Trained on emotion detection (joy, sadness, anger, etc.)")
    print("     NOT trained for sentiment analysis")
    print("     Labels may not align with sentiment")
    print("    After fine-tuning: will learn financial sentiment")
    print()
    print("=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print()
    print("BEFORE Fine-tuning (current results):")
    print("   ¢ FinBERT should be most accurate (already trained on finance)")
    print("   ¢ DistilBERT may be reasonable (general sentiment)")
    print("   ¢ ELECTRA will be off (emotions  sentiment)")
    print()
    print("AFTER Fine-tuning on FinancialPhraseBank:")
    print("   ¢ All 3 models will have same 3 labels (neg, neu, pos)")
    print("   ¢ All will understand financial terminology")
    print("   ¢ We can compare which ARCHITECTURE learns best:")
    print("     - FinBERT: Domain specialist, may improve slightly")
    print("     - DistilBERT: Smaller/faster, may lose 1-2% accuracy")
    print("     - ELECTRA: Often outperforms BERT, may be best")
    print()
    print(" NEXT STEPS:")
    print("   1. Review pretrained_comparison.csv")
    print("   2. Note FinBERT's accuracy (your baseline)")
    print("   3. Train all 3 models on FinancialPhraseBank")
    print("   4. Test again with test_finetuned_models.py")
    print("   5. Compare improvements!")
    print()
    
    # ============================================
    # SAVE DETAILED ANALYSIS REPORT
    # ============================================
    report_path = "pretrained_models_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" PRE-TRAINED MODELS ANALYSIS REPORT\n")
        f.write(" Baseline Performance Before Fine-tuning\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Dataset: FinancialPhraseBank (balanced sample)\n")
        f.write(f"Number of Test Sentences: {len(df)}\n")
        f.write(f"Results CSV: {output_file}\n\n")
        # Metrics
        f.write("="*80 + "\n")
        f.write("METRICS (Precision, Recall, F1, Accuracy)\n")
        f.write("="*80 + "\n\n")
        for model, y_pred in zip(['FinBERT', 'DistilBERT', 'ELECTRA'], [y_pred_finbert, y_pred_distilbert, y_pred_electra]):
            f.write(f"{model}:\n")
            f.write(classification_report(y_true, y_pred, digits=3, zero_division=0))
            f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n\n")
        # Winner
        f.write(f" WINNER: {winner} with {accuracies[winner]:.1f}% accuracy\n\n")
        # Per-class Accuracy
        f.write("="*80 + "\n")
        f.write("PER-CLASS ACCURACY BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        for label in ['positive', 'negative', 'neutral']:
            class_df = df[df['true_label'] == label]
            if len(class_df) > 0:
                finbert_class_acc = class_df['finbert_correct'].sum() / len(class_df) * 100
                distilbert_class_acc = class_df['distilbert_correct'].sum() / len(class_df) * 100
                electra_class_acc = class_df['electra_correct'].sum() / len(class_df) * 100
                f.write(f"{label.upper()} ({len(class_df)} samples):\n")
                f.write(f"   FinBERT:    {finbert_class_acc:5.1f}%\n")
                f.write(f"   DistilBERT: {distilbert_class_acc:5.1f}%\n")
                f.write(f"   ELECTRA:    {electra_class_acc:5.1f}%\n\n")
        # Confusion Matrix
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX - PREDICTION BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        for model in ['finbert', 'distilbert', 'electra']:
            f.write(f"{model.upper()}:\n")
            pred_col = f'{model}_prediction'
            for true_label in ['positive', 'negative', 'neutral']:
                class_preds = df[df['true_label'] == true_label][pred_col].value_counts()
                if len(class_preds) > 0:
                    pred_str = ", ".join([f"{k}:{v}" for k, v in dict(class_preds).items()])
                    f.write(f"   True {true_label:8s}  {pred_str}\n")
            f.write("\n")
        
        # Model Characteristics
        f.write("="*80 + "\n")
        f.write("MODEL CHARACTERISTICS\n")
        f.write("="*80 + "\n\n")
        f.write("1. FinBERT (ProsusAI/finbert)\n")
        f.write("   - Pre-trained on financial text (already domain-specific)\n")
        f.write("   - Has 3 sentiment labels: positive, negative, neutral\n")
        f.write("   - Expected to perform best at baseline\n")
        f.write("   - 110M parameters\n\n")
        
        f.write("2. DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)\n")
        f.write("   - Pre-trained on movie reviews (SST-2 dataset)\n")
        f.write("   - Only 2 labels: POSITIVE, NEGATIVE (no neutral)\n")
        f.write("   - 66M parameters (40% smaller than BERT)\n")
        f.write("   - 60% faster inference than BERT\n")
        f.write("   - Expected to struggle with neutral and financial context\n\n")
        
        f.write("3. ELECTRA (google/electra-base-discriminator)\n")
        f.write("   - Pre-trained on emotion detection task\n")
        f.write("   - Not specifically trained for sentiment\n")
        f.write("   - 110M parameters\n")
        f.write("   - Sample-efficient training (learns faster)\n")
        f.write("   - Expected to perform poorly at baseline\n\n")
        
        # Key Insights
        f.write("="*80 + "\n")
        f.write("KEY INSIGHTS & EXPECTATIONS\n")
        f.write("="*80 + "\n\n")
        f.write("BEFORE Fine-tuning (Current Results):\n")
        f.write("   ¢ FinBERT should dominate (already financial sentiment trained)\n")
        f.write("   ¢ DistilBERT may have reasonable accuracy on positive/negative\n")
        f.write("   ¢ DistilBERT will struggle with neutral (no neutral label)\n")
        f.write("   ¢ ELECTRA will be inconsistent (emotion  sentiment)\n\n")
        
        f.write("AFTER Fine-tuning on FinancialPhraseBank:\n")
        f.write("   ¢ All models will learn the same 3 labels\n")
        f.write("   ¢ All will understand financial terminology\n")
        f.write("   ¢ Architecture differences will determine performance:\n")
        f.write("     - FinBERT: May improve 3-5% (already optimized)\n")
        f.write("     - DistilBERT: Should improve 10-15% (learns neutral + finance)\n")
        f.write("     - ELECTRA: Should improve 20-30% (learns task from scratch)\n\n")
        
        f.write("Expected Post-Training Accuracy:\n")
        f.write("   ¢ FinBERT: 80-83% (current baseline + refinement)\n")
        f.write("   ¢ DistilBERT: 79-81% (trade-off: speed vs accuracy)\n")
        f.write("   ¢ ELECTRA: 82-84% (best architecture for this task)\n\n")
        
        # Sample Predictions
        f.write("="*80 + "\n")
        f.write("SAMPLE PREDICTIONS (First 10 sentences)\n")
        f.write("="*80 + "\n\n")
        for i, row in df.head(10).iterrows():
            sentence = row['sentence'][:100] + "..." if len(row['sentence']) > 100 else row['sentence']
            f.write(f"Sentence {i+1}: {sentence}\n")
            f.write(f"   True Label:      {row['true_label']}\n")
            f.write(f"   FinBERT:         {row['finbert_prediction']} {'' if row['finbert_correct'] else ''}\n")
            f.write(f"   DistilBERT:      {row['distilbert_prediction']} {'' if row['distilbert_correct'] else ''}\n")
            f.write(f"   ELECTRA:         {row['electra_prediction']} {'' if row['electra_correct'] else ''}\n\n")
        
        # Next Steps
        f.write("="*80 + "\n")
        f.write("RECOMMENDED NEXT STEPS\n")
        f.write("="*80 + "\n\n")
        f.write("1. Review this analysis and the CSV file (pretrained_comparison.csv)\n")
        f.write("2. Note FinBERT's baseline accuracy as the target to beat\n")
        f.write("3. Train DistilBERT using: colab_training_distilbert.py\n")
        f.write("4. Train ELECTRA using: colab_training_electra.py\n")
        f.write("5. Create test_finetuned_models.py to evaluate trained models\n")
        f.write("6. Compare before/after improvements for each model\n")
        f.write("7. Select final model based on accuracy/speed trade-off\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f" Detailed analysis report saved to: {report_path}")
    print()
    print("=" * 70)
    print(" TESTING COMPLETE")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
