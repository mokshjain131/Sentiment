from __future__ import annotations
from typing import Iterable

# Map categorical label + probs to a numeric sentiment score [-1,1]
LABEL_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

def label_to_score(label: str) -> float:
    return LABEL_SCORE.get(label, 0.0)


def aggregate_article_prob(probs: dict[str, float]) -> float:
    # Expect keys: negative, neutral, positive
    return probs.get("positive",0) - probs.get("negative",0)


def decay_weight(index: int, half_life: int = 50) -> float:
    # Simple exponential decay by position (newest index=0)
    import math
    return 0.5 ** (index / max(1, half_life))
