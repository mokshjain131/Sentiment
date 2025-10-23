from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from ..input.schemas import FetchParams
from ..data.loaders.news_api_loader import fetch_and_store


def main():
    parser = argparse.ArgumentParser("news sentiment ingestion")
    parser.add_argument("--company", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--keywords", default="")
    parser.add_argument("--days", type=int, default=1, help="Look-back window in whole days (24h multiples)")
    parser.add_argument("--max", type=int, default=300)
    parser.add_argument("--verbose", action="store_true", help="Print debug info (query, dates, counts)")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="Output file format (default: parquet)")
    args = parser.parse_args()

    # Use precise UTC timestamps instead of date-only (which truncates to 00:00 and can miss current-day articles)
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=args.days)
    # ISO 8601 strings acceptable by NewsAPI
    to_iso = to_dt.isoformat(timespec="seconds")
    from_iso = from_dt.isoformat(timespec="seconds")

    params = FetchParams(
        company=args.company,
        ticker=args.ticker,
        keywords=[k.strip() for k in args.keywords.split(",") if k.strip()],
        from_date=from_iso,
        to_date=to_iso,
        max_articles=args.max,
    )
    count = fetch_and_store(params, verbose=args.verbose, file_format=args.format)
    if args.verbose:
        print(f"Finished ingestion. Articles stored: {count}")
    else:
        print(f"Ingested {count} articles.")


if __name__ == "__main__":  # pragma: no cover
    main()
