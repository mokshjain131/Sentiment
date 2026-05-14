from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime
from pydantic import field_validator

class FetchParams(BaseModel):
    company: str
    ticker: str
    industry: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    from_date: Optional[str | datetime | date] = None
    to_date: Optional[str | datetime | date] = None
    language: str = "en"
    max_articles: int = 500

    @field_validator("from_date", "to_date", mode="before")
    @classmethod
    def normalize_date(cls, v):
        if v is None:
            return v
        if isinstance(v, datetime):
            return v.isoformat(timespec="seconds")
        if isinstance(v, date):
            return datetime(v.year, v.month, v.day).isoformat() + "Z"
        if isinstance(v, str):
            return v
        return str(v)

class Article(BaseModel):
    hash: str
    source: str
    title: str
    description: Optional[str]
    content: Optional[str]
    published_at: datetime
    url: str
    language: str
    tickers: List[str]
    raw: dict
