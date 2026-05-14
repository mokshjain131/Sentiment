from __future__ import annotations
import argparse
from ..aggregation.report_builder import build_reports

def main():
    ap = argparse.ArgumentParser("aggregate sentiment metrics")
    ap.add_argument("--dataset", default="data/datasets/weak_labeled.parquet")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="Output file format (default: parquet)")
    args = ap.parse_args()
    results = build_reports(args.dataset, args.ticker, args.out, window=args.window, file_format=args.format)
    print("Wrote reports:")
    for k,v in results.items():
     print(f" {k}: {v}")

if __name__ == "__main__": # pragma: no cover
    main()
