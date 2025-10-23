from __future__ import annotations
import argparse
from ..models.finetune import fine_tune, evaluate_model

def main():
    ap = argparse.ArgumentParser("fine-tune sentiment model on weak labels")
    ap.add_argument("--dataset", default="data/datasets/weak_labeled.parquet")
    ap.add_argument("--model_name", default="ProsusAI/finbert")
    ap.add_argument("--out", default="models/finetuned")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--label-col", default="weak_label", help="Column name for labels (use 'label' for FinancialPhraseBank)")
    ap.add_argument("--eval", action="store_true", help="Only evaluate an existing model directory")
    args = ap.parse_args()

    if args.eval:
        metrics = evaluate_model(args.out, args.dataset)
        print(metrics)
        return

    model_dir = fine_tune(args.model_name, args.dataset, output_dir=args.out, epochs=args.epochs, batch_size=args.batch, lr=args.lr, label_col=args.label_col)
    metrics = evaluate_model(model_dir, args.dataset)
    print("Training complete. Metrics:")
    print(metrics)

if __name__ == "__main__":  # pragma: no cover
    main()
