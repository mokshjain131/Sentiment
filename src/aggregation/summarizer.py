from __future__ import annotations
import pandas as pd
from .scoring import label_to_score, aggregate_article_prob, decay_weight
from datetime import datetime


def prepare_articles(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure needed columns exist
    required = {"published_at", "weak_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for aggregation: {missing}")
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df['weak_label'].map(label_to_score)
    return df


def daily_ticker_aggregate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df[df['tickers'].apply(lambda x: ticker in x if isinstance(x, list) else False)].copy()
    if df.empty:
        return pd.DataFrame(columns=['date','ticker','mean_score','pct_pos','pct_neg','article_count','sentiment_volatility'])
    df['date'] = pd.to_datetime(df['published_at']).dt.date
    grp = df.groupby('date')
    rows = []
    for dt, g in grp:
        scores = g['sentiment_score']
        mean_score = scores.mean()
        article_count = len(g)
        pct_pos = (g['weak_label'] == 'positive').mean()
        pct_neg = (g['weak_label'] == 'negative').mean()
        sentiment_volatility = scores.std(ddof=0) if article_count > 1 else 0.0
        rows.append({
            'date': dt,
            'ticker': ticker,
            'mean_score': mean_score,
            'pct_pos': pct_pos,
            'pct_neg': pct_neg,
            'article_count': article_count,
            'sentiment_volatility': sentiment_volatility,
        })
    return pd.DataFrame(rows).sort_values('date')


def rolling_trend(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['rolling_mean'] = df['mean_score'].rolling(window, min_periods=1).mean()
    df['rolling_z'] = (df['mean_score'] - df['rolling_mean']) / (df['mean_score'].rolling(window, min_periods=2).std(ddof=0) + 1e-6)
    return df


def detect_alerts(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    if 'rolling_z' not in df.columns:
        return pd.DataFrame(columns=['date','ticker','alert_type','z_value'])
    alerts = df[abs(df['rolling_z']) >= z_threshold]
    return alerts.assign(alert_type=lambda d: d['rolling_z'].apply(lambda z: 'spike_positive' if z>0 else 'spike_negative'))[['date','ticker','alert_type','rolling_z']].rename(columns={'rolling_z':'z_value'})
