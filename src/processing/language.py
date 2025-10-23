from __future__ import annotations
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"
