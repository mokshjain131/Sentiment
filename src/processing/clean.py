from __future__ import annotations
import re
import html

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
 text = html.unescape(text)
 # Remove HTML tags
 text = re.sub(r"<[^>]+>", " ", text)
 # Normalize quotes
 text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
 # Remove common boilerplate prefixes
 text = re.sub(r"\(Reuters\) ?- ?", "", text, flags=re.I)
 text = re.sub(r"\(Bloomberg\) ?--? ?", "", text, flags=re.I)
 # Collapse whitespace
 text = _WS.sub(" ", text).strip()
 return text
