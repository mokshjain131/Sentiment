from __future__ import annotations
import time
import hashlib
from typing import Iterator, Optional
import requests
from dataclasses import dataclass
from functools import lru_cache
import os
import yaml
try:
    from dotenv import load_dotenv # type: ignore
except Exception: # pragma: no cover
    load_dotenv = None # type: ignore

# Load .env early (if python-dotenv installed). Safe no-op if not present.
if load_dotenv is not None:
    # Allow project root .env discovery
    load_dotenv() # relies on current working directory when process started


@dataclass
class AppConfig:
    newsapi_key: str
    requests_per_minute: int = 25
    page_size: int = 100
    base_url: str = "https://newsapi.org/v2/everything"


@lru_cache
def load_config(path: Optional[str] = None) -> AppConfig:
    data = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    key = os.getenv("NEWSAPI_KEY") or data.get("newsapi_key")
    # Strip wrapping quotes if user added them in .env
    if key and ((key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'"))):
        key = key[1:-1]
    if not key:
        cwd = os.getcwd()
        raise ValueError(
            "Missing NEWSAPI_KEY. Set environment variable, or add to .env, or provide in config file. "
            f"CWD={cwd}. Checked env + optional YAML path={path or 'N/A'}."
        )
    return AppConfig(
        newsapi_key=key,
        requests_per_minute=data.get("requests_per_minute", 25),
        page_size=data.get("page_size", 100),
        base_url=data.get("base_url", "https://newsapi.org/v2/everything"),
    )


def build_query(company: str, ticker: str, keywords: list[str]) -> str:
    parts = {company}
    if ticker:
        parts.add(ticker)
    parts.update(k.strip() for k in keywords if k.strip())
    formatted = []
    for p in parts:
        if " " in p:
            formatted.append(f'"{p}"')
        else:
            formatted.append(p)
    return " OR ".join(formatted)


def hash_article(title: str, source: str, published: str) -> str:
    base = f"{title.lower().strip()}|{source.lower()}|{published[:10]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


class NewsAPIClient:
    def __init__(self, config: Optional[AppConfig] = None, session: Optional[requests.Session] = None):
        self.cfg = config or load_config(None)
        self.session = session or requests.Session()
        self.rate_interval = 60 / self.cfg.requests_per_minute

    def fetch(self, query: str, from_date=None, to_date=None, page_size: int = 100, max_pages: int = 5) -> Iterator[dict]:
        for page in range(1, max_pages + 1):
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": self.cfg.newsapi_key,
                "pageSize": page_size,
                "page": page,
            }
            if from_date:
                params["from"] = str(from_date)
            if to_date:
                params["to"] = str(to_date)
            resp = self.session.get(self.cfg.base_url, params=params, timeout=20)
            # Rate limit -> backoff and retry same page
            if resp.status_code == 429:
                time.sleep(5)
                continue
            # Free tier limitations often respond with 426 on additional pages
            if resp.status_code == 426:
                print("[newsapi] Received 426 Upgrade Required. Likely exceeded free plan capabilities (e.g. requesting additional pages or parameters). Returning collected results.")
                if page == 1:
                    # No first page content, raise so user can see
                    resp.raise_for_status()
                break
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                print(f"[newsapi] HTTP error {resp.status_code}: {e}. Stopping pagination.")
                break
            data = resp.json()
            articles = data.get("articles", [])
            for a in articles:
                yield a
            # Stop if fewer than a full page was returned
            if len(articles) < page_size:
                break
            time.sleep(self.rate_interval)
