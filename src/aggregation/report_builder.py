from __future__ import annotations
import pandas as pd
from pathlib import Path
from .summarizer import daily_ticker_aggregate, rolling_trend, detect_alerts, prepare_articles


def build_reports(labeled_path: str, ticker: str, out_dir: str = 'reports', window: int = 7, file_format: str = "parquet") -> dict:
    # Auto-detect input format
    labeled_path_obj = Path(labeled_path)
    if labeled_path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(labeled_path)
        # CSV doesn't preserve list types - parse tickers column if it's a string
        if 'tickers' in df.columns and df['tickers'].dtype == 'object':
            import ast
            df['tickers'] = df['tickers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    else:
        df = pd.read_parquet(labeled_path)

    # Prepare articles (adds sentiment_score if missing)
    df = prepare_articles(df)

    daily = daily_ticker_aggregate(df, ticker)
    daily = rolling_trend(daily, window=window)
    alerts = detect_alerts(daily)
    out = Path(out_dir) / ticker.lower()
    out.mkdir(parents=True, exist_ok=True)

    # Save in specified format
    if file_format.lower() == "csv":
        daily_fp = out / 'daily.csv'
        alerts_fp = out / 'alerts.csv'
        daily.to_csv(daily_fp, index=False)
        alerts.to_csv(alerts_fp, index=False)
    else:
        daily_fp = out / 'daily.parquet'
        alerts_fp = out / 'alerts.parquet'
        daily.to_parquet(daily_fp, index=False)
        alerts.to_parquet(alerts_fp, index=False)

    return {'daily': str(daily_fp), 'alerts': str(alerts_fp)}
