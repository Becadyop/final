"""
Echo Feeling - Public API
Provides the `analyze` function as the primary entry-point for the library.

Usage:
    from echo_feeling.api import analyze, analyze_batch, product_dashboard

    result = analyze("This product is great! 😍")
    print(result["label"])      # positive
    print(result["confidence"]) # 0.93
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .engine import SentimentEngine

# Singleton engine ─ lazily initialised
_ENGINE: SentimentEngine | None = None


def _get_engine(model_dir: str | None = None) -> SentimentEngine:
    global _ENGINE
    if _ENGINE is None or (model_dir and str(_ENGINE.model_dir) != model_dir):
        _ENGINE = SentimentEngine(model_dir=model_dir or str(
            Path(__file__).parent.parent / "models"
        ))
        try:
            _ENGINE.load()
        except FileNotFoundError:
            # Models not yet trained – return engine in stub mode
            pass
    return _ENGINE


# ── Public API ───────────────────────────────────────────────────────────────

def analyze(
    text: str,
    sticker: str = "neutral",
    model_dir: str | None = None,
) -> dict:
    """
    Analyse the sentiment of a single review.

    Parameters
    ----------
    text        : raw review text (may include emojis)
    sticker     : associated sticker sentiment label –
                  one of 'positive', 'negative', 'neutral'
    model_dir   : optional path to trained model artefacts

    Returns
    -------
    dict
        label       – 'positive' | 'negative' | 'neutral' | 'suspicious'
        confidence  – float in [0, 1]
        scores      – per-class probability dict
        emoji_score – aggregate emoji sentiment in [-1, 1]

    Examples
    --------
    >>> from echo_feeling.api import analyze
    >>> analyze("Excellent product, very happy! 😊👍")
    {'label': 'positive', 'confidence': 0.94, 'scores': {...}, 'emoji_score': 0.85}
    """
    engine = _get_engine(model_dir)
    if not engine.is_loaded:
        # Fallback rule-based prediction when model is not available
        return _rule_based_fallback(text)
    return engine.predict(text, sticker_label=sticker)


def analyze_batch(
    texts: list[str],
    stickers: list[str] | None = None,
    model_dir: str | None = None,
) -> list[dict]:
    """
    Analyse a list of reviews in batch.

    Parameters
    ----------
    texts    : list of raw review strings
    stickers : optional list of sticker labels (same length as texts)
    model_dir: optional path to trained model artefacts

    Returns
    -------
    list of result dicts (same schema as `analyze`)
    """
    engine = _get_engine(model_dir)
    if not engine.is_loaded:
        return [_rule_based_fallback(t) for t in texts]
    return engine.predict_batch(texts, sticker_labels=stickers)


def product_dashboard(
    reviews: list[str],
    stickers: list[str] | None = None,
    model_dir: str | None = None,
) -> dict:
    """
    Return aggregated sentiment statistics for all reviews of a product.
    Designed for consumption by the admin panel dashboard.

    Returns
    -------
    dict
        total_reviews      – int
        counts             – {label: count}
        percentages        – {label: percentage}
        avg_confidence     – float
        avg_emoji_score    – float
        flagged_suspicious – list of review indices flagged as suspicious
    """
    engine = _get_engine(model_dir)
    if not engine.is_loaded:
        results = [_rule_based_fallback(r) for r in reviews]
        return _aggregate(results, len(reviews))
    return engine.product_summary(reviews, sticker_labels=stickers)


# ── Rule-based fallback (no trained model) ───────────────────────────────────

_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "best",
    "fantastic", "perfect", "happy", "wonderful", "superb", "recommend",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "worst", "hate", "poor", "horrible",
    "disappointing", "broken", "useless", "waste", "fake", "fraud",
}


def _rule_based_fallback(text: str) -> dict:
    """Simple lexicon-based fallback when model artefacts are absent."""
    words = set(text.lower().split())
    pos   = len(words & _POSITIVE_WORDS)
    neg   = len(words & _NEGATIVE_WORDS)

    if pos > neg:
        label, conf = "positive", min(0.5 + pos * 0.1, 0.9)
    elif neg > pos:
        label, conf = "negative", min(0.5 + neg * 0.1, 0.9)
    else:
        label, conf = "neutral", 0.5

    return {"label": label, "confidence": conf, "scores": {label: conf}, "emoji_score": 0.0}


def _aggregate(results: list[dict], total: int) -> dict:
    from collections import Counter
    counts = Counter(r["label"] for r in results)
    return {
        "total_reviews"      : total,
        "counts"             : dict(counts),
        "percentages"        : {k: round(v / total * 100, 1) for k, v in counts.items()},
        "avg_confidence"     : round(sum(r["confidence"] for r in results) / max(total, 1), 4),
        "avg_emoji_score"    : 0.0,
        "flagged_suspicious" : [],
    }
