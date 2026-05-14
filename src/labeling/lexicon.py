from __future__ import annotations
import re
from typing import Iterable

# Minimal financial lexicon (placeholder). Extend later.
POSITIVE_WORDS = {"growth", "beat", "surge", "record", "upgrade", "profit"}
NEGATIVE_WORDS = {"loss", "downgrade", "decline", "lawsuit", "probe", "miss"}

WORD_RE = re.compile(r"[A-Za-z']+")


def score_lexicon(text: str) -> float:
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    if not tokens:
     return 0.0
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    total = pos + neg
    if total == 0:
     return 0.0
    return (pos - neg) / total # in [-1,1]


def label_from_score(s: float) -> str:
    if s > 0.2:
     return "positive"
    if s < -0.2:
     return "negative"
    return "neutral"


def lexicon_label(text: str) -> dict:
    s = score_lexicon(text)
    return {"label": label_from_score(s), "score": s}
