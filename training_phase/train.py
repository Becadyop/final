"""
Echo Feeling - Training Pipeline
Trains SVM, Random Forest, and Feedforward Neural Network classifiers on the
multimodal fused feature vectors and saves the best model.

Usage:
    python train.py --data_path data/reviews.csv --sticker_root data/stickers
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessor import TextPreprocessor, EmojiPreprocessor, StickerPreprocessor
from feature_extractor import (
    TextFeatureExtractor, EmojiFeatureExtractor,
    StickerFeatureExtractor, MultimodalFusion,
)

# ── Label constants ──────────────────────────────────────────────────────────
LABEL_MAP    = {"positive": 2, "neutral": 1, "negative": 0, "suspicious": -1}
LABEL_NAMES  = ["negative", "neutral", "positive"]
MODEL_DIR    = Path("models")


# ────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Expects a CSV with at least:
      - 'review'   : raw review text (may contain emojis)
      - 'label'    : positive / negative / neutral / suspicious
      - 'sticker'  : sticker sentiment label (positive/negative/neutral/none)
    """
    df = pd.read_csv(csv_path)
    required = {"review", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if "sticker" not in df.columns:
        df["sticker"] = "neutral"
    df = df[df["label"].isin(["positive", "negative", "neutral", "suspicious"])].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Loaded {len(df)} samples. Label distribution:\n{df['label'].value_counts()}")
    return df


# ────────────────────────────────────────────────────────────────────────────
# Build feature matrix
# ────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    text_extractor: TextFeatureExtractor,
    emoji_extractor: EmojiFeatureExtractor,
    sticker_extractor: StickerFeatureExtractor,
    fusion: MultimodalFusion,
    fit: bool = True,
) -> np.ndarray:

    tp = TextPreprocessor()
    ep = EmojiPreprocessor()
    sp = StickerPreprocessor()

    # Text
    clean_texts = tp.batch_clean([ep.strip_emojis(r) for r in df["review"]])
    if fit:
        text_extractor.fit(clean_texts)
    text_feats = text_extractor.transform(clean_texts)

    # Emoji
    emoji_df = ep.build_emoji_dataframe(df["review"].tolist())
    if fit:
        emoji_feats = emoji_extractor.fit_transform(emoji_df)
    else:
        emoji_feats = emoji_extractor.transform(emoji_df)

    # Sticker
    sticker_weights = sp.encode_labels(df["sticker"].tolist())
    sticker_feats   = sticker_extractor.transform(sticker_weights)

    # Fuse
    X = fusion.fuse(text_feats, emoji_feats, sticker_feats, fit=fit)
    return X


# ────────────────────────────────────────────────────────────────────────────
# Model definitions
# ────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam",
            max_iter=300, early_stopping=True,
            random_state=42,
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
    """Run k-fold cross-validation and return averaged metrics."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    scores  = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "accuracy" : float(np.mean(scores["test_accuracy"])),
        "precision": float(np.mean(scores["test_precision_macro"])),
        "recall"   : float(np.mean(scores["test_recall_macro"])),
        "f1"       : float(np.mean(scores["test_f1_macro"])),
    }


def confusion_analysis(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ────────────────────────────────────────────────────────────────────────────
# Main training routine
# ────────────────────────────────────────────────────────────────────────────

def train(data_path: str, sticker_root: str | None, use_bert: bool, output_dir: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset(data_path)

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])

    # Initialise extractors
    text_ext    = TextFeatureExtractor(use_bert=use_bert)
    emoji_ext   = EmojiFeatureExtractor()
    sticker_ext = StickerFeatureExtractor()
    fusion      = MultimodalFusion()

    print("[INFO] Building feature matrix…")
    X = build_features(df, text_ext, emoji_ext, sticker_ext, fusion, fit=True)
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # Save fitted extractors
    text_ext.save(str(out / "text_vectorizers.pkl"))
    emoji_ext.save(str(out / "emoji_scaler.pkl"))
    fusion.save(str(out / "fusion_scaler.pkl"))
    joblib.dump(le, str(out / "label_encoder.pkl"))

    # Train & evaluate all models
    results = {}
    best_f1, best_name, best_model = -1.0, None, None

    for name, model in get_models().items():
        print(f"\n[TRAIN] {name}…")
        metrics = evaluate_model(model, X, y, k=5)
        results[name] = metrics
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")

        # Fit final model on full training set
        model.fit(X, y)
        joblib.dump(model, str(out / f"model_{name}.pkl"))

        if metrics["f1"] > best_f1:
            best_f1   = metrics["f1"]
            best_name = name
            best_model = model

    # Detailed confusion analysis for best model
    print(f"\n[INFO] Best model: {best_name} (F1={best_f1:.4f})")
    confusion_analysis(best_model, X, y)

    # Save results & best model pointer
    with open(str(out / "training_results.json"), "w") as f:
        json.dump({"models": results, "best": best_name}, f, indent=2)

    # Copy best model as canonical model
    import shutil
    shutil.copy(str(out / f"model_{best_name}.pkl"), str(out / "model_best.pkl"))

    print(f"\n[DONE] All artefacts saved to '{out}'")
    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Echo Feeling multimodal sentiment model")
    parser.add_argument("--data_path",    required=True,  help="Path to reviews CSV file")
    parser.add_argument("--sticker_root", default=None,   help="Root folder for sticker PNG dataset")
    parser.add_argument("--use_bert",     action="store_true", help="Include BERT embeddings (slow)")
    parser.add_argument("--output_dir",   default="models", help="Directory to save trained artefacts")
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        sticker_root=args.sticker_root,
        use_bert=args.use_bert,
        output_dir=args.output_dir,
    )
