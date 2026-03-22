"""
Echo Feeling
============
A multimodal sentiment analysis library for e-commerce platforms.

Analyses customer reviews integrating text, emoji, and sticker modalities
using BoW, TF-IDF, BERT embeddings, and ensemble ML classifiers.

Quick start
-----------
>>> from echo_feeling.api import analyze
>>> result = analyze("Absolutely love this product! 😍👍")
>>> result["label"]
'positive'

>>> from echo_feeling.api import product_dashboard
>>> summary = product_dashboard(["Great!", "Terrible 😡", "It's okay"])
>>> summary["counts"]
{'positive': 1, 'negative': 1, 'neutral': 1}
"""

from .api import analyze, analyze_batch, product_dashboard
from .engine import SentimentEngine

__all__ = ["analyze", "analyze_batch", "product_dashboard", "SentimentEngine"]
__version__ = "2.0.0"
__author__  = "Swathi Lekshmi SS, Romin Varghese, R Arun, Ivin Issac"
