"""
Echo Feeling - Unit Tests
Run with: pytest tests/ -v
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Make local imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training_phase"))

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        from preprocessor import TextPreprocessor
        self.tp = TextPreprocessor()

    def test_basic_clean(self):
        result = self.tp.clean("Hello World! This is GREAT.")
        assert isinstance(result, str)
        assert result == result.lower()

    def test_removes_urls(self):
        result = self.tp.clean("Visit https://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_removes_html(self):
        result = self.tp.clean("<b>Bold</b> text here")
        assert "<b>" not in result
        assert "bold" in result or "text" in result

    def test_handles_empty_string(self):
        assert self.tp.clean("") == ""

    def test_handles_non_string(self):
        assert self.tp.clean(None) == ""
        assert self.tp.clean(123) == ""

    def test_batch_clean(self):
        texts = ["Hello world!", "Bad product 😡", "Okay I guess"]
        results = self.tp.batch_clean(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestEmojiPreprocessor:
    def setup_method(self):
        from preprocessor import EmojiPreprocessor
        self.ep = EmojiPreprocessor()

    def test_positive_emoji_score(self):
        score = self.ep.score("I love this! 😍❤️👍")
        assert score > 0

    def test_negative_emoji_score(self):
        score = self.ep.score("Terrible! 😡👎💔")
        assert score < 0

    def test_no_emoji_score_zero(self):
        score = self.ep.score("This is a plain text review")
        assert score == 0.0

    def test_strip_emojis(self):
        cleaned = self.ep.strip_emojis("Great product 😍 very good 👍")
        assert "😍" not in cleaned
        assert "👍" not in cleaned
        assert "Great" in cleaned or "great" in cleaned

    def test_build_emoji_dataframe_shape(self):
        texts = ["I love it 😊", "Terrible 😡", "Just okay"]
        df = self.ep.build_emoji_dataframe(texts)
        assert df.shape == (3, 2)
        assert "emoji_count" in df.columns
        assert "emoji_score" in df.columns


class TestStickerPreprocessor:
    def setup_method(self):
        from preprocessor import StickerPreprocessor
        self.sp = StickerPreprocessor()

    def test_encode_labels(self):
        weights = self.sp.encode_labels(["positive", "negative", "neutral"])
        assert list(weights) == [1.0, -1.0, 0.0]

    def test_sentiment_weight_positive(self):
        assert self.sp.get_sentiment_weight("positive") == 1.0

    def test_sentiment_weight_negative(self):
        assert self.sp.get_sentiment_weight("negative") == -1.0

    def test_sentiment_weight_neutral(self):
        assert self.sp.get_sentiment_weight("neutral") == 0.0

    def test_load_labels_no_root(self):
        df = self.sp.load_labels()
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTextFeatureExtractor:
    def setup_method(self):
        from feature_extractor import TextFeatureExtractor
        self.extractor = TextFeatureExtractor(use_bert=False, max_features=100)
        self.texts = [
            "great product love it",
            "terrible waste of money",
            "decent quality average",
            "excellent fast delivery",
            "broken product unhappy",
        ]
        self.extractor.fit(self.texts)

    def test_bow_shape(self):
        feats = self.extractor.bow(self.texts)
        assert feats.shape[0] == len(self.texts)
        assert feats.shape[1] <= 100

    def test_tfidf_shape(self):
        feats = self.extractor.tfidf(self.texts)
        assert feats.shape[0] == len(self.texts)
        assert feats.shape[1] <= 100

    def test_transform_shape(self):
        feats = self.extractor.transform(self.texts)
        assert feats.shape[0] == len(self.texts)
        assert feats.dtype == np.float32

    def test_transform_single(self):
        feats = self.extractor.transform(["single review here"])
        assert feats.shape[0] == 1


class TestEmojiFeatureExtractor:
    def setup_method(self):
        from feature_extractor import EmojiFeatureExtractor
        self.extractor = EmojiFeatureExtractor()

    def test_fit_transform_shape(self):
        df = pd.DataFrame({"emoji_count": [1, 2, 0], "emoji_score": [0.8, -0.5, 0.0]})
        feats = self.extractor.fit_transform(df)
        assert feats.shape == (3, 2)

    def test_transform_after_fit(self):
        df = pd.DataFrame({"emoji_count": [1, 0], "emoji_score": [0.5, 0.0]})
        self.extractor.fit_transform(df)
        feats = self.extractor.transform(df)
        assert feats.shape == (2, 2)


class TestStickerFeatureExtractor:
    def setup_method(self):
        from feature_extractor import StickerFeatureExtractor
        self.extractor = StickerFeatureExtractor()

    def test_transform_shape(self):
        weights = np.array([1.0, -1.0, 0.0])
        feats   = self.extractor.transform(weights)
        assert feats.shape == (3, 1)

    def test_fine_tune_weight(self):
        self.extractor.fine_tune_weight(2.0)
        weights = np.array([1.0])
        feats   = self.extractor.transform(weights)
        assert feats[0, 0] == pytest.approx(2.0)


class TestMultimodalFusion:
    def setup_method(self):
        from feature_extractor import MultimodalFusion
        self.fusion = MultimodalFusion()

    def test_fuse_shape(self):
        text_f    = np.random.rand(5, 50).astype(np.float32)
        emoji_f   = np.random.rand(5, 2).astype(np.float32)
        sticker_f = np.random.rand(5, 1).astype(np.float32)
        fused = self.fusion.fuse(text_f, emoji_f, sticker_f, fit=True)
        assert fused.shape == (5, 53)

    def test_fuse_dtype(self):
        text_f    = np.random.rand(3, 20).astype(np.float32)
        emoji_f   = np.random.rand(3, 2).astype(np.float32)
        sticker_f = np.random.rand(3, 1).astype(np.float32)
        fused = self.fusion.fuse(text_f, emoji_f, sticker_f, fit=True)
        assert fused.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Public API Tests (rule-based fallback — no trained models needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def setup_method(self):
        from echo_feeling.api import analyze, analyze_batch, product_dashboard
        self.analyze           = analyze
        self.analyze_batch     = analyze_batch
        self.product_dashboard = product_dashboard

    def test_analyze_returns_dict(self):
        result = self.analyze("This is a great product!")
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result

    def test_analyze_positive(self):
        result = self.analyze("This is the best product ever, love it!")
        assert result["label"] == "positive"

    def test_analyze_negative(self):
        result = self.analyze("Terrible waste of money, horrible product")
        assert result["label"] == "negative"

    def test_analyze_confidence_range(self):
        result = self.analyze("Decent product, okay quality")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_analyze_batch_length(self):
        texts = ["Great!", "Terrible!", "Okay."]
        results = self.analyze_batch(texts)
        assert len(results) == 3

    def test_analyze_batch_each_has_label(self):
        texts = ["Great quality!", "Bad experience", "Average product"]
        for result in self.analyze_batch(texts):
            assert "label" in result
            assert result["label"] in ("positive", "negative", "neutral", "suspicious")

    def test_product_dashboard_keys(self):
        reviews = ["Great!", "Terrible!", "Okay."]
        summary = self.product_dashboard(reviews)
        assert "total_reviews" in summary
        assert "counts" in summary
        assert "percentages" in summary

    def test_product_dashboard_total(self):
        reviews = ["Great!", "Terrible!", "Okay."]
        summary = self.product_dashboard(reviews)
        assert summary["total_reviews"] == 3

    def test_product_dashboard_percentages_sum(self):
        reviews = ["Great!", "Terrible!", "Okay.", "Love it!", "Worst ever"]
        summary = self.product_dashboard(reviews)
        total_pct = sum(summary["percentages"].values())
        assert abs(total_pct - 100.0) < 1.0  # allow rounding

    def test_analyze_with_emojis(self):
        result = self.analyze("Love this product so much! 😍❤️👍")
        assert result["label"] in ("positive", "negative", "neutral", "suspicious")
        assert "emoji_score" in result

    def test_analyze_sticker_parameter(self):
        result = self.analyze("Decent product", sticker="positive")
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: data generation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGenerator:
    def test_generates_correct_count(self):
        sys.path.insert(0, str(ROOT))
        from generate_sample_data import generate_dataset
        rows = generate_dataset(n=100)
        assert len(rows) == 100

    def test_all_rows_have_required_fields(self):
        from generate_sample_data import generate_dataset
        for row in generate_dataset(n=50):
            assert "review" in row
            assert "label"  in row
            assert "sticker" in row

    def test_labels_are_valid(self):
        from generate_sample_data import generate_dataset
        valid = {"positive", "negative", "neutral", "suspicious"}
        for row in generate_dataset(n=100):
            assert row["label"] in valid

    def test_stickers_are_valid(self):
        from generate_sample_data import generate_dataset
        valid = {"positive", "negative", "neutral"}
        for row in generate_dataset(n=100):
            assert row["sticker"] in valid
