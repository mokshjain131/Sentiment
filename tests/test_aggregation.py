import pandas as pd
from datetime import datetime, timedelta
from src.aggregation.summarizer import daily_ticker_aggregate, rolling_trend, detect_alerts

def _mock_df():
    base = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    rows = []
    labels = ['positive','negative','neutral','positive','negative','positive']
    for i, lab in enumerate(labels):
        rows.append({
            'published_at': base - timedelta(days=(len(labels)-i)),
            'weak_label': lab,
            'tickers': ['AAPL'],
            'sentiment_score': 1 if lab=='positive' else (-1 if lab=='negative' else 0)
        })
    return pd.DataFrame(rows)

def test_daily_and_alerts():
    df = _mock_df()
    daily = daily_ticker_aggregate(df, 'AAPL')
    assert not daily.empty
    with_trend = rolling_trend(daily, window=3)
    alerts = detect_alerts(with_trend, z_threshold=10)  # high threshold -> usually none
    assert set(['date','ticker','mean_score']).issubset(daily.columns)
    assert alerts.shape[1] == 4  # columns subset
