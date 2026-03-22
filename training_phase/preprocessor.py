"""
Echo Feeling - Preprocessing Module
Handles text normalization, emoji standardization, and sticker labeling.
"""

import re
import os
import json
import string
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path

# ── Optional heavy deps ─────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"):
        nltk.download(pkg, quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import emoji as emoji_lib
    EMOJI_LIB_AVAILABLE = True
except ImportError:
    EMOJI_LIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Emoji sentiment lexicon (Unicode → polarity score) ──────────────────────
EMOJI_SENTIMENT_MAP: dict[str, float] = {
    "😀": 0.9, "😃": 0.9, "😄": 0.85, "😁": 0.85, "😆": 0.8,
    "😊": 0.9, "🥰": 0.95, "😍": 0.9, "🤩": 0.9, "😎": 0.75,
    "👍": 0.8,  "❤️": 0.95, "💯": 0.9, "🎉": 0.85, "✅": 0.7,
    "😢": -0.8, "😭": -0.9, "😡": -0.9, "🤬": -0.95, "💔": -0.85,
    "👎": -0.8, "😤": -0.75, "😠": -0.85, "🤮": -0.9, "😞": -0.7,
    "😐": 0.0,  "😑": 0.0,  "🤔": 0.0,  "😶": 0.0,  "😏": 0.1,
}

# ── Sticker sentiment folder names ─────────────────────────────────────────
STICKER_LABELS = {"positive": 1, "negative": -1, "neutral": 0}


class TextPreprocessor:
    """Normalizes raw review text: tokenize → lowercase → remove stop-words → lemmatize."""

    def __init__(self):
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))
        else:
            self.lemmatizer = None
            self.stop_words = set()

    def clean(self, text: str) -> str:
        """Full preprocessing pipeline for a single text string."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Separate emojis from words (keep for emoji processing later)
        text = re.sub(r"[^\w\s\U00010000-\U0010ffff]", " ", text, flags=re.UNICODE)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Tokenize
        tokens = text.split() if not NLTK_AVAILABLE else word_tokenize(text)
        # Remove stop-words and lemmatize
        if NLTK_AVAILABLE:
            tokens = [
                self.lemmatizer.lemmatize(t)
                for t in tokens
                if t not in self.stop_words and len(t) > 1
            ]
        else:
            tokens = [t for t in tokens if len(t) > 1]
        return " ".join(tokens)

    def batch_clean(self, texts: list[str]) -> list[str]:
        return [self.clean(t) for t in texts]


class EmojiPreprocessor:
    """
    Extracts emojis from text and maps them to sentiment scores.
    Returns both the cleaned text (emojis stripped) and an aggregate emoji score.
    """

    def extract_emojis(self, text: str) -> list[str]:
        """Return a list of emoji characters found in text."""
        if EMOJI_LIB_AVAILABLE:
            return [ch for ch in text if ch in emoji_lib.EMOJI_DATA]
        # Fallback: detect via unicode ranges
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\u2600-\u26FF\u2700-\u27BF]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.findall(text)

    def score(self, text: str) -> float:
        """Aggregate sentiment score from emojis in text. Returns value in [-1, 1]."""
        emojis = self.extract_emojis(text)
        if not emojis:
            return 0.0
        scores = [EMOJI_SENTIMENT_MAP.get(e, 0.0) for e in emojis]
        return float(np.mean(scores))

    def strip_emojis(self, text: str) -> str:
        """Remove emoji characters from text."""
        if EMOJI_LIB_AVAILABLE:
            return emoji_lib.replace_emoji(text, replace="")
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\u2600-\u26FF\u2700-\u27BF]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    def build_emoji_dataframe(self, texts: list[str]) -> pd.DataFrame:
        """Build a DataFrame with emoji scores for each review."""
        rows = []
        for text in texts:
            emojis = self.extract_emojis(text)
            score = self.score(text)
            rows.append({"emoji_count": len(emojis), "emoji_score": score})
        return pd.DataFrame(rows)


class StickerPreprocessor:
    """
    Loads sticker PNG images organized into positive/negative/neutral folders,
    encodes labels, and returns a simple numerical representation per sticker.
    """

    def __init__(self, sticker_root: str | None = None):
        self.sticker_root = Path(sticker_root) if sticker_root else None
        self.label_map = STICKER_LABELS  # positive→1, negative→-1, neutral→0

    def load_labels(self) -> pd.DataFrame:
        """Scan sticker folders and return a DataFrame of (filename, label, score)."""
        if self.sticker_root is None or not self.sticker_root.exists():
            return pd.DataFrame(columns=["filename", "label", "score"])
        rows = []
        for folder, score in self.label_map.items():
            folder_path = self.sticker_root / folder
            if not folder_path.exists():
                continue
            for img_file in folder_path.glob("*.png"):
                rows.append({
                    "filename": str(img_file),
                    "label": folder,
                    "score": score,
                })
        return pd.DataFrame(rows)

    def get_sentiment_weight(self, sticker_label: str) -> float:
        """Convert a sticker label string to its numeric sentiment weight."""
        return float(self.label_map.get(sticker_label, 0.0))

    @staticmethod
    def encode_labels(labels: list[str]) -> np.ndarray:
        """Encode categorical sticker labels to numeric weights."""
        enc = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        return np.array([enc.get(l, 0.0) for l in labels], dtype=np.float32)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tp = TextPreprocessor()
    ep = EmojiPreprocessor()

    sample = "This product is absolutely amazing! 😍❤️ Highly recommend it 👍"
    print("Original :", sample)
    print("Cleaned  :", tp.clean(ep.strip_emojis(sample)))
    print("Emoji score:", ep.score(sample))
