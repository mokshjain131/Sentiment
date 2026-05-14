"""
Test Trained Models (After Fine-tuning)
Compares performance of fine-tuned FinBERT, DistilBERT, and ELECTRA models

Usage:
    python test_trained_models.py

Output:
    trained_models_comparison.csv - Predictions from all 3 trained models
    trained_models_analysis.txt - Detailed analysis report
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

def load_trained_finbert():
    """Load trained FinBERT from local models directory"""
    print("¦ Loading trained FinBERT...")
    model_path = Path("models/Colab/FinBERT_financial")

    if not model_path.exists():
        print(f"    Trained FinBERT not found at {model_path}")
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        # Get label mapping from config
        if hasattr(model.config, 'id2label'):
            label_map = model.config.id2label
        else:
            label_map = {0: "negative", 1: "neutral", 2: "positive"}  # Standard sentiment mapping

        print(f"    Trained FinBERT loaded (labels: {list(label_map.values())})")
        return tokenizer, model, label_map
    except Exception as e:
        print(f"    Error loading trained FinBERT: {e}")
        return None, None, None

def load_trained_distilbert():
    """Load trained DistilBERT from local models directory"""
    print("¦ Loading trained DistilBERT...")
    model_path = Path("models/Colab/distilbert_financial")

    if not model_path.exists():
        print(f"    Trained DistilBERT not found at {model_path}")
        return None

    try:
        # Load as pipeline for consistency
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=str(model_path),
            tokenizer=str(model_path)
        )
        print("    Trained DistilBERT loaded (3-class sentiment)")
        return sentiment_pipeline
    except Exception as e:
        print(f"    Error loading trained DistilBERT: {e}")
        return None

def load_trained_electra():
    """Load trained ELECTRA from local models directory"""
    print("¦ Loading trained ELECTRA...")
    model_path = Path("models/Colab/electra_financial")

    if not model_path.exists():
        print(f"    Trained ELECTRA not found at {model_path}")
        return None

    try:
        # Load as pipeline for consistency
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=str(model_path),
            tokenizer=str(model_path)
        )
        print("    Trained ELECTRA loaded (3-class sentiment)")
        return sentiment_pipeline
    except Exception as e:
        print(f"    Error loading trained ELECTRA: {e}")
        return None

def predict_trained_finbert(text, tokenizer, model, label_map):
    """Predict with trained FinBERT"""
    if tokenizer is None or model is None:
        return "N/A", 0.0, None

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            confidence = probs[0][pred_idx].item()
            all_probs = probs[0].tolist()

        return label_map[pred_idx], confidence, all_probs
    except Exception as e:
        print(f"    Error predicting with FinBERT: {e}")
        return "ERROR", 0.0, None

def predict_trained_distilbert(text, pipeline):
    """Predict with trained DistilBERT"""
    if pipeline is None:
        return "N/A", 0.0, None

    try:
        result = pipeline(text, truncation=True, max_length=512)[0]
        label = result['label'].lower()
        confidence = result['score']
        return label, confidence, None
    except Exception as e:
        print(f"    Error predicting with DistilBERT: {e}")
        return "ERROR", 0.0, None

def predict_trained_electra(text, pipeline):
    """Predict with trained ELECTRA"""
    if pipeline is None:
        return "N/A", 0.0, None

    try:
        result = pipeline(text, truncation=True, max_length=512)[0]
        label = result['label'].lower()
        confidence = result['score']
        return label, confidence, None
    except Exception as e:
        print(f"    Error predicting with ELECTRA: {e}")
        return "ERROR", 0.0, None

def main():
    print("=" * 70)
    print("§ª TESTING TRAINED MODELS (After Fine-tuning)")
    print("=" * 70)
    print()
    print("This script tests 3 FINE-TUNED models:")
    print("1. FinBERT - Fine-tuned on FinancialPhraseBank")
    print("2. DistilBERT - Fine-tuned on FinancialPhraseBank")
    print("3. ELECTRA - Fine-tuned on FinancialPhraseBank")
    print()
    print(f"Testing on {len(TEST_SENTENCES)} financial sentences...")
    print()
    print("=" * 70)
    print()

    # Load trained models from local directory
    print("¥ Loading trained models from models/Colab/...\n")
    finbert_tokenizer, finbert_model, finbert_labels = load_trained_finbert()
    distilbert_pipeline = load_trained_distilbert()
    electra_pipeline = load_trained_electra()

    print()
    print(" Model loading complete!")
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
            finbert_label, finbert_conf, finbert_probs = predict_trained_finbert(
                sentence, finbert_tokenizer, finbert_model, finbert_labels
            )

            # DistilBERT prediction
            distilbert_label, distilbert_conf, _ = predict_trained_distilbert(
                sentence, distilbert_pipeline
            )

            # ELECTRA prediction
            electra_label, electra_conf, _ = predict_trained_electra(
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
                'electra_correct': electra_label == true_label
            }

            # Add probability details for FinBERT (has 3 classes)
            if finbert_probs and len(finbert_probs) >= 3:
                result['finbert_prob_negative'] = round(finbert_probs[0], 4)
                result['finbert_prob_neutral'] = round(finbert_probs[1], 4)
                result['finbert_prob_positive'] = round(finbert_probs[2], 4)

            results.append(result)

            # Display predictions
            finbert_mark = "" if finbert_label == true_label else ""
            distilbert_mark = "" if distilbert_label == true_label else ""
            electra_mark = "" if electra_label == true_label else ""

            print(f"   FinBERT:    {finbert_label:8s} ({finbert_conf*100:.1f}%) {finbert_mark}")
            print(f"   DistilBERT: {distilbert_label:8s} ({distilbert_conf*100:.1f}%) {distilbert_mark}")
            print(f"   ELECTRA:    {electra_label:8s} ({electra_conf*100:.1f}%) {electra_mark}")
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
    output_file = "trained_models_comparison.csv"
    df.to_csv(output_file, index=False)

    # ============================================
    # ACCURACY COMPARISON REPORT
    # ============================================

    print("=" * 70)
    print(" ACCURACY COMPARISON REPORT - TRAINED MODELS")
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
    print(f" WINNER (Fine-tuned Models): {winner} with {accuracies[winner]:.1f}% accuracy")
    print("=" * 70)
    print()

    print(f" Full results saved to: {output_file}")
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

    # Agreement analysis
    finbert_pos_neg = df[df['finbert_prediction'].isin(['positive', 'negative', 'neutral'])]
    if len(finbert_pos_neg) > 0:
        agreement_fd = 0
        agreement_fe = 0
        agreement_de = 0

        for _, row in finbert_pos_neg.iterrows():
            if row['finbert_prediction'] == row['distilbert_prediction']:
                agreement_fd += 1
            if row['finbert_prediction'] == row['electra_prediction']:
                agreement_fe += 1
            if row['distilbert_prediction'] == row['electra_prediction']:
                agreement_de += 1

        agreement_fd_pct = (agreement_fd / len(finbert_pos_neg)) * 100
        agreement_fe_pct = (agreement_fe / len(finbert_pos_neg)) * 100
        agreement_de_pct = (agreement_de / len(finbert_pos_neg)) * 100

        print("Model Agreement Analysis:")
        print(f"   FinBERT vs DistilBERT: {agreement_fd}/{len(finbert_pos_neg)} ({agreement_fd_pct:.1f}%)")
        print(f"   FinBERT vs ELECTRA:    {agreement_fe}/{len(finbert_pos_neg)} ({agreement_fe_pct:.1f}%)")
        print(f"   DistilBERT vs ELECTRA: {agreement_de}/{len(finbert_pos_neg)} ({agreement_de_pct:.1f}%)")
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
    print("1£  FinBERT (Fine-tuned):")
    print("    Fine-tuned on FinancialPhraseBank dataset")
    print("    Should show significant improvement over pre-trained")
    print("    Expected accuracy: 80-85%")
    print()
    print("2£  DistilBERT (Fine-tuned):")
    print("    Learned neutral class + financial terminology")
    print("    Smaller model, faster inference")
    print("    Expected accuracy: 79-82%")
    print()
    print("3£  ELECTRA (Fine-tuned):")
    print("    Learned sentiment task from scratch")
    print("    Sample-efficient training")
    print("    Expected accuracy: 82-85%")
    print()
    print("=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print()
    print("AFTER Fine-tuning:")
    print("   ¢ All models now have proper 3-class sentiment")
    print("   ¢ All understand financial terminology")
    print("   ¢ Compare against pre-trained baseline!")
    print()
    print(" NEXT STEPS:")
    print("   1. Compare with pretrained_comparison.csv")
    print("   2. Calculate improvement percentages")
    print("   3. Choose best model for production")
    print("   4. Deploy the winner!")
    print()
    print("=" * 70)
    print(" TESTING COMPLETE")
    print("=" * 70)
    print()

    # ============================================
    # SAVE DETAILED ANALYSIS REPORT
    # ============================================

    report_path = "trained_models_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" TRAINED MODELS ANALYSIS REPORT\n")
        f.write(" Performance After Fine-tuning on FinancialPhraseBank\n")
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
        f.write("1. FinBERT (Fine-tuned)\n")
        f.write("   - Pre-trained on financial text, fine-tuned on FinancialPhraseBank\n")
        f.write("   - 110M parameters\n")
        f.write("   - Domain specialist for financial sentiment\n\n")

        f.write("2. DistilBERT (Fine-tuned)\n")
        f.write("   - Pre-trained on general text, fine-tuned on FinancialPhraseBank\n")
        f.write("   - 66M parameters (40% smaller than BERT)\n")
        f.write("   - 60% faster inference\n")
        f.write("   - Learned neutral class and financial terminology\n\n")

        f.write("3. ELECTRA (Fine-tuned)\n")
        f.write("   - Pre-trained on emotion detection, fine-tuned on FinancialPhraseBank\n")
        f.write("   - 110M parameters\n")
        f.write("   - Sample-efficient training\n")
        f.write("   - Learned sentiment task from scratch\n\n")

        # Key Insights
        f.write("="*80 + "\n")
        f.write("KEY INSIGHTS & COMPARISON TO PRE-TRAINED\n")
        f.write("="*80 + "\n\n")
        f.write("BEFORE vs AFTER Fine-tuning:\n")
        f.write("   ¢ Pre-trained: FinBERT best (already financial), others inconsistent\n")
        f.write("   ¢ Fine-tuned: All models should show major improvements\n")
        f.write("   ¢ Expected gains: 20-40% accuracy improvement\n\n")

        f.write("Architecture Comparison:\n")
        f.write("   ¢ FinBERT: May improve 3-8% (refinement of existing knowledge)\n")
        f.write("   ¢ DistilBERT: Should improve 25-35% (learned neutral + finance)\n")
        f.write("   ¢ ELECTRA: Should improve 30-40% (learned task from scratch)\n\n")

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
        f.write("1. Compare trained_models_comparison.csv with pretrained_comparison.csv\n")
        f.write("2. Calculate improvement percentages for each model\n")
        f.write("3. Analyze which architecture benefited most from fine-tuning\n")
        f.write("4. Select the best model for production deployment\n")
        f.write("5. Consider model size vs accuracy trade-offs\n\n")

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