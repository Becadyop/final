# text_preprocess_pipeline.py
# -----------------------------
# Preprocess Amazon reviews dataset for text sentiment analysis (3 classes)

import pandas as pd
import torch
import os
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\data\text\amazon_reviews.csv",
    encoding="utf-8"
)

print("First 5 rows of dataset:")
print(df.head())

# -------------------------------
# 2. Clean text (basic normalization)
# -------------------------------
df["comment_clean"] = df["verified_reviews"].astype(str).str.lower().str.strip()

# -------------------------------
# 3. Sentiment labels (map rating → 3 classes)
# -------------------------------
def map_rating_to_label(rating):
    if rating <= 2:
        return 0   # negative
    elif rating == 3:
        return 1   # neutral
    else:
        return 2   # positive

df["label_numeric"] = df["rating"].apply(map_rating_to_label)
y = df["label_numeric"].values

print("Unique sentiment labels:", set(y))
print("Counts per class:", pd.Series(y).value_counts())

# -------------------------------
# 4. Tokenizer
# -------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer vocab size:", len(tokenizer))

# -------------------------------
# 5. Train/Test split
# -------------------------------
X = df["comment_clean"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tokenize train/test sets
encoded_train = tokenizer(
    X_train, padding=True, truncation=True, max_length=64, return_tensors="pt"
)
encoded_test = tokenizer(
    X_test, padding=True, truncation=True, max_length=64, return_tensors="pt"
)

print("\nTrain/Test split complete.")
print("Train size:", encoded_train['input_ids'].shape[0], "Test size:", encoded_test['input_ids'].shape[0])

# -------------------------------
# 6. Save outputs to .pt files
# -------------------------------
save_dir = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_text"
os.makedirs(save_dir, exist_ok=True)

torch.save(encoded_train, os.path.join(save_dir, "text_train_tokens.pt"))
torch.save(encoded_test, os.path.join(save_dir, "text_test_tokens.pt"))
torch.save(torch.tensor(y_train), os.path.join(save_dir, "text_train_labels.pt"))
torch.save(torch.tensor(y_test), os.path.join(save_dir, "text_test_labels.pt"))

print("\n✅ Preprocessed text outputs saved to:", save_dir)

# -------------------------------
# 7. Save human-readable CSV
# -------------------------------
csv_path = os.path.join(save_dir, "text_dataset_info.csv")

df_out = pd.DataFrame({
    "original_review": df["verified_reviews"],
    "comment_clean": df["comment_clean"],
    "sentiment_label": df["label_numeric"].map({0: "negative", 1: "neutral", 2: "positive"}),
    "label_numeric": df["label_numeric"]
})

df_out.to_csv(csv_path, index=False, encoding="utf-8")
print("\n✅ Human-readable CSV saved to:", csv_path)

# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Text preprocessing pipeline complete!")
print("Dataset shape:", df.shape)
