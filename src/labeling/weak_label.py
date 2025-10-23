from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from .lexicon import lexicon_label

@dataclass
class WeakLabelResult:
    text: str
    finbert_label: str | None
    finbert_probs: dict | None
    lex_label: str
    lex_score: float
    final_label: str


def combine_labels(finbert_label: str | None, finbert_probs: dict | None, lex_label: str, lex_score: float) -> str:
    # Simple precedence: if FinBERT exists and max prob > 0.6 use it; else lexicon
    if finbert_label and finbert_probs:
        max_prob = max(finbert_probs.values())
        if max_prob >= 0.6:
            return finbert_label
    return lex_label


def weak_label_batch(texts: List[str], finbert=None) -> List[WeakLabelResult]:
    finbert_results: List[Dict] | None = None
    if finbert is not None:
        finbert_results = finbert.predict(texts)
    out: List[WeakLabelResult] = []
    for idx, text in enumerate(texts):
        lex = lexicon_label(text)
        fb_label = None
        fb_probs = None
        if finbert_results:
            fb_label = finbert_results[idx]["label"]
            fb_probs = finbert_results[idx]["probs"]
        final = combine_labels(fb_label, fb_probs, lex["label"], lex["score"])
        out.append(WeakLabelResult(
            text=text,
            finbert_label=fb_label,
            finbert_probs=fb_probs,
            lex_label=lex["label"],
            lex_score=lex["score"],
            final_label=final,
        ))
    return out
