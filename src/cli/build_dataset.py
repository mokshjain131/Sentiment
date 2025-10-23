from __future__ import annotations
import argparse
import glob
from pathlib import Path
import pandas as pd
from .main import main as _unused  # noqa: F401
from ..labeling.weak_label import weak_label_batch
try:
    from ..models.finbert.inference import FinBERTSentiment
except Exception:  # pragma: no cover
    FinBERTSentiment = None


def build_dataset(processed_dir: str = "data/processed", out_path: str = "data/datasets/weak_labeled.parquet", use_finbert: bool = False, sample: int | None = None, file_format: str = "parquet"):
    # Support both parquet and csv input files
    if file_format.lower() == "csv":
        paths = glob.glob(f"{processed_dir}/*/*.csv")
    else:
        paths = glob.glob(f"{processed_dir}/*/*.parquet")
    
    if not paths:
        raise SystemExit(f"No processed {file_format} files found. Run ingestion first with --format {file_format}")
    
    # Load files based on format
    frames = []
    for p in paths:
        if file_format.lower() == "csv":
            frames.append(pd.read_csv(p))
        else:
            frames.append(pd.read_parquet(p))
    
    df = pd.concat(frames, ignore_index=True)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)
    texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).str.strip()
    finbert = None
    if use_finbert and FinBERTSentiment is not None:
        finbert = FinBERTSentiment()
    results = weak_label_batch(texts.tolist(), finbert=finbert)
    df_labels = pd.DataFrame([{
        "hash": df.iloc[i]["hash"],
        "weak_label": r.final_label,
        "lex_label": r.lex_label,
        "lex_score": r.lex_score,
        "finbert_label": r.finbert_label,
        "finbert_probs": r.finbert_probs,
    } for i, r in enumerate(results)])
    merged = df.merge(df_labels, on="hash", how="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in the same format as input or use output extension
    output_ext = out_path.suffix.lower()
    if output_ext == ".csv":
        merged.to_csv(out_path, index=False)
    else:
        merged.to_parquet(out_path, index=False)
    
    return str(out_path)


def cli():
    parser = argparse.ArgumentParser("build weak labeled dataset")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out", default="data/datasets/weak_labeled.parquet")
    parser.add_argument("--finbert", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="Input file format (default: parquet)")
    args = parser.parse_args()
    path = build_dataset(args.processed_dir, args.out, use_finbert=args.finbert, sample=args.sample, file_format=args.format)
    print(f"Wrote weak-labeled dataset to {path}")

if __name__ == "__main__":  # pragma: no cover
    cli()
