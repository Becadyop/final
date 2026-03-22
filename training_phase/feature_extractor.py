"""
Echo Feeling - Feature Extraction Module
Extracts BoW, TF-IDF, BERT embeddings, emoji features, and sticker features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class TextFeatureExtractor:
    """
    Extracts Bag-of-Words, TF-IDF, and optionally BERT embeddings from text.
    All vectorizers are fitted on training data and reused at inference.
    """

    def __init__(self, use_bert: bool = False, max_features: int = 5000):
        self.use_bert = use_bert and BERT_AVAILABLE
        self.max_features = max_features
        self.bow_vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.scaler = StandardScaler(with_mean=False)  # sparse-safe

        if self.use_bert:
            print("[INFO] Loading BERT model (bert-base-uncased)…")
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
            self.bert_model.eval()
        else:
            self.bert_tokenizer = None
            self.bert_model = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "TextFeatureExtractor":
        self.bow_vectorizer.fit(texts)
        self.tfidf_vectorizer.fit(texts)
        return self

    # ── Transform ────────────────────────────────────────────────────────────

    def bow(self, texts: list[str]) -> np.ndarray:
        return self.bow_vectorizer.transform(texts).toarray()

    def tfidf(self, texts: list[str]) -> np.ndarray:
        return self.tfidf_vectorizer.transform(texts).toarray()

    def bert_embeddings(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return [CLS] BERT embeddings (768-dim) for each text."""
        if not self.use_bert:
            return np.zeros((len(texts), 1))
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = self.bert_tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.bert_model(**encoded)
            cls = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls)
        return np.vstack(embeddings)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Return concatenated BoW + TF-IDF (+ optional BERT) features."""
        bow_feats   = self.bow(texts)
        tfidf_feats = self.tfidf(texts)
        parts = [bow_feats, tfidf_feats]
        if self.use_bert:
            parts.append(self.bert_embeddings(texts))
        return np.hstack(parts).astype(np.float32)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(
            {"bow": self.bow_vectorizer, "tfidf": self.tfidf_vectorizer},
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.bow_vectorizer  = data["bow"]
        self.tfidf_vectorizer = data["tfidf"]


class EmojiFeatureExtractor:
    """
    Converts the emoji sentiment dataframe produced by EmojiPreprocessor
    into a normalized numpy feature matrix.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, emoji_df: pd.DataFrame) -> np.ndarray:
        feats = emoji_df[["emoji_count", "emoji_score"]].values.astype(np.float32)
        out = self.scaler.fit_transform(feats)
        self._fitted = True
        return out

    def transform(self, emoji_df: pd.DataFrame) -> np.ndarray:
        feats = emoji_df[["emoji_count", "emoji_score"]].values.astype(np.float32)
        if self._fitted:
            return self.scaler.transform(feats)
        return feats

    def save(self, path: str) -> None:
        joblib.dump(self.scaler, path)

    def load(self, path: str) -> None:
        self.scaler  = joblib.load(path)
        self._fitted = True


class StickerFeatureExtractor:
    """
    Converts encoded sticker labels (float weights) into a single-column feature.
    Also supports optional fine-tuning weight adjustment.
    """

    def __init__(self):
        self.weight_scale = 1.0  # fine-tunable multiplier

    def transform(self, sticker_weights: np.ndarray) -> np.ndarray:
        """Return a (N, 1) feature array from sticker sentiment weights."""
        return (sticker_weights * self.weight_scale).reshape(-1, 1).astype(np.float32)

    def fine_tune_weight(self, scale: float) -> None:
        """Adjust the contribution of the sticker modality."""
        self.weight_scale = scale


class MultimodalFusion:
    """
    Concatenates normalized text, emoji, and sticker feature vectors
    into a single unified representation for classification.
    """

    def __init__(self):
        self.final_scaler = StandardScaler()
        self._fitted = False

    def fuse(
        self,
        text_feats: np.ndarray,
        emoji_feats: np.ndarray,
        sticker_feats: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Concatenate and optionally scale all modality features.

        Parameters
        ----------
        text_feats    : (N, D1)
        emoji_feats   : (N, 2)
        sticker_feats : (N, 1)
        fit           : if True, fit the scaler on this data
        """
        fused = np.hstack([text_feats, emoji_feats, sticker_feats])
        if fit:
            out = self.final_scaler.fit_transform(fused)
            self._fitted = True
        elif self._fitted:
            out = self.final_scaler.transform(fused)
        else:
            out = fused
        return out.astype(np.float32)

    def save(self, path: str) -> None:
        joblib.dump(self.final_scaler, path)

    def load(self, path: str) -> None:
        self.final_scaler = joblib.load(path)
        self._fitted       = True
