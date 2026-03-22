"""
Echo Feeling - Core Sentiment Engine
Loads trained artefacts and runs the full multimodal inference pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# ── Local imports (works both as package and from training_phase) ────────────
_HERE = Path(__file__).parent
_TRAINING = _HERE.parent / "training_phase"

import sys
for _p in [str(_HERE), str(_TRAINING)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessor import TextPreprocessor, EmojiPreprocessor, StickerPreprocessor
from feature_extractor import (
    TextFeatureExtractor, EmojiFeatureExtractor,
    StickerFeatureExtractor, MultimodalFusion,
)

# ── Sentiment label constants ────────────────────────────────────────────────
LABEL_POSITIVE   = "positive"
LABEL_NEGATIVE   = "negative"
LABEL_NEUTRAL    = "neutral"
LABEL_SUSPICIOUS = "suspicious"

# Default model directory (overrideable via env)
DEFAULT_MODEL_DIR = os.environ.get(
    "ECHO_FEELING_MODEL_DIR",
    str(Path(__file__).parent.parent / "models"),
)


class SentimentEngine:
    """
    Multimodal sentiment analysis engine.

    Loads:
      • text_vectorizers.pkl  – BoW + TF-IDF vectorisers
      • emoji_scaler.pkl      – StandardScaler for emoji features
      • fusion_scaler.pkl     – StandardScaler for fused feature vector
      • label_encoder.pkl     – sklearn LabelEncoder
      • model_best.pkl        – best trained classifier

    Inference accepts any combination of text, emoji (embedded in text),
    and a sticker_label string ("positive" / "negative" / "neutral").
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._loaded   = False

        self.text_extractor    = TextFeatureExtractor(use_bert=False)
        self.emoji_extractor   = EmojiFeatureExtractor()
        self.sticker_extractor = StickerFeatureExtractor()
        self.fusion            = MultimodalFusion()
        self.label_encoder     = None
        self.model             = None

        self.text_preprocessor   = TextPreprocessor()
        self.emoji_preprocessor  = EmojiPreprocessor()
        self.sticker_preprocessor = StickerPreprocessor()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "SentimentEngine":
        """Load all artefacts from model_dir. Raises FileNotFoundError if missing."""
        md = self.model_dir
        self.text_extractor.load(str(md / "text_vectorizers.pkl"))
        self.emoji_extractor.load(str(md / "emoji_scaler.pkl"))
        self.fusion.load(str(md / "fusion_scaler.pkl"))
        self.label_encoder = joblib.load(str(md / "label_encoder.pkl"))
        self.model         = joblib.load(str(md / "model_best.pkl"))
        self._loaded = True
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ─────────────────────────────────────────────────────────────

    def _build_single_feature(self, review: str, sticker_label: str) -> np.ndarray:
        """Build a (1, D) feature vector for a single review."""
        # Text
        clean = self.text_preprocessor.clean(
            self.emoji_preprocessor.strip_emojis(review)
        )
        text_feats = self.text_extractor.transform([clean])  # (1, D1)

        # Emoji
        ep_df      = self.emoji_preprocessor.build_emoji_dataframe([review])
        emoji_feats = self.emoji_extractor.transform(ep_df)  # (1, 2)

        # Sticker
        sticker_weight = self.sticker_preprocessor.encode_labels([sticker_label])
        sticker_feats  = self.sticker_extractor.transform(sticker_weight)  # (1, 1)

        return self.fusion.fuse(text_feats, emoji_feats, sticker_feats, fit=False)

    def predict(self, review: str, sticker_label: str = "neutral") -> dict:
        """
        Analyse a single review.

        Parameters
        ----------
        review        : raw review string (may include emojis)
        sticker_label : one of 'positive', 'negative', 'neutral'

        Returns
        -------
        dict with keys: label, confidence, scores, emoji_score
        """
        if not self._loaded:
            self.load()

        X = self._build_single_feature(review, sticker_label)

        label_idx  = self.model.predict(X)[0]
        label      = self.label_encoder.inverse_transform([label_idx])[0]

        # Probability scores per class
        if hasattr(self.model, "predict_proba"):
            proba  = self.model.predict_proba(X)[0]
            scores = {
                cls: float(p)
                for cls, p in zip(self.label_encoder.classes_, proba)
            }
            confidence = float(max(proba))
        else:
            scores     = {label: 1.0}
            confidence = 1.0

        emoji_score = self.emoji_preprocessor.score(review)

        return {
            "label"      : label,
            "confidence" : confidence,
            "scores"     : scores,
            "emoji_score": emoji_score,
        }

    def predict_batch(
        self,
        reviews: list[str],
        sticker_labels: list[str] | None = None,
    ) -> list[dict]:
        """Analyse a batch of reviews."""
        if not self._loaded:
            self.load()
        if sticker_labels is None:
            sticker_labels = ["neutral"] * len(reviews)
        return [
            self.predict(r, s)
            for r, s in zip(reviews, sticker_labels)
        ]

    def product_summary(self, reviews: list[str], sticker_labels: list[str] | None = None) -> dict:
        """
        Return aggregated sentiment statistics for a product's review set.
        Mirrors the admin-panel dashboard output described in the paper.
        """
        results = self.predict_batch(reviews, sticker_labels)
        counts  = {LABEL_POSITIVE: 0, LABEL_NEGATIVE: 0, LABEL_NEUTRAL: 0, LABEL_SUSPICIOUS: 0}
        for r in results:
            lbl = r["label"]
            if lbl in counts:
                counts[lbl] += 1
            else:
                counts[LABEL_NEUTRAL] += 1

        total = len(results)
        return {
            "total_reviews"      : total,
            "counts"             : counts,
            "percentages"        : {k: round(v / total * 100, 1) if total else 0 for k, v in counts.items()},
            "avg_confidence"     : round(float(np.mean([r["confidence"] for r in results])), 4),
            "avg_emoji_score"    : round(float(np.mean([r["emoji_score"] for r in results])), 4),
            "flagged_suspicious" : [i for i, r in enumerate(results) if r["label"] == LABEL_SUSPICIOUS],
        }
