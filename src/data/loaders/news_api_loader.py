from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
from ..loaders import * # noqa: F401
from ...input.schemas import FetchParams, Article
from ...services.news_api_client import NewsAPIClient, build_query, hash_article
from ...processing.clean import clean_text
from ...processing.language import detect_lang


def fetch_and_store(params: FetchParams, out_dir: str = "data", client: NewsAPIClient | None = None, verbose: bool = False, file_format: str = "parquet") -> int:
    client = client or NewsAPIClient()
    query = build_query(params.company, params.ticker, params.keywords)
    if verbose:
        print(f"[ingest] Query: {query}")
        print(f"[ingest] Window: {params.from_date} -> {params.to_date}")
        print(f"[ingest] Output format: {file_format}")

    raw_records = []
    seen: set[str] = set()
    max_pages = params.max_articles // client.cfg.page_size + 1

    for raw in client.fetch(query, params.from_date, params.to_date, page_size=client.cfg.page_size, max_pages=max_pages):
        title = raw.get("title") or ""
        source = (raw.get("source") or {}).get("name", "unknown")
        ts = raw.get("publishedAt") or ""
        h = hash_article(title, source, ts)
        if h in seen:
            continue
        seen.add(h)
        small_text = " ".join(filter(None, [title, raw.get("description") or ""]))
        lang = detect_lang(small_text)
        if lang != params.language:
            continue
        combined = " ".join(filter(None, [title, raw.get("description") or "", raw.get("content") or ""]))
        cleaned = clean_text(combined)
        tickers = [params.ticker] if params.ticker.lower() in cleaned.lower() else []
        try:
            published_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        art = Article(
            hash=h,
            source=source,
            title=title,
            description=raw.get("description"),
            content=raw.get("content"),
            published_at=published_dt,
            url=raw.get("url"),
            language=lang,
            tickers=tickers,
            raw=raw,
        )
        raw_records.append(art.model_dump())
        if len(raw_records) >= params.max_articles:
            break

    if not raw_records:
        if verbose:
            print("[ingest] No articles matched criteria. Potential reasons: \n"
                  " - Query too restrictive (try fewer OR terms)\n"
                  " - Time window has no results (expand --days)\n"
                  " - API key plan restrictions or rate limit (check headers)\n"
                  " - NewsAPI returns only recent major stories, shorten query")
        return 0

    df = pd.DataFrame(raw_records)
    date_part = df.published_at.dt.strftime("%Y%m%d").mode()[0]
    processed_path = Path(out_dir) / "processed" / date_part
    processed_path.mkdir(parents=True, exist_ok=True)

    # Support both parquet and csv formats
    if file_format.lower() == "csv":
        fname = f"{params.ticker.lower()}_{date_part}.csv"
        df.to_csv(processed_path / fname, index=False)
    else:
        fname = f"{params.ticker.lower()}_{date_part}.parquet"
        df.to_parquet(processed_path / fname, index=False)

    if verbose:
        print(f"[ingest] Wrote {len(df)} rows to {processed_path / fname}")
    return len(df)
