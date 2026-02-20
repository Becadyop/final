# emoji_preprocess_pipeline.py
# -----------------------------
# Preprocess Emoji Sentiment Data v1.0 for sentiment analysis

import pandas as pd
import torch
import os
import emoji
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\data\emoji\Emoji_Sentiment_Data_v1.0.csv",
    encoding="utf-8"
)

print("First 5 rows of dataset:")
print(df.head())

# -------------------------------
# 2. Normalize emojis
# -------------------------------
df["emoji_clean"] = df["Emoji"].astype(str).apply(lambda e: emoji.demojize(e).strip())

# -------------------------------
# 3. Derive sentiment labels
# -------------------------------
def get_sentiment(row):
    counts = {"negative": row["Negative"], "neutral": row["Neutral"], "positive": row["Positive"]}
    return max(counts, key=counts.get)

df["sentiment_label"] = df.apply(get_sentiment, axis=1)

# Map to numeric labels (0=negative, 1=neutral, 2=positive)
label_map = {"negative": 0, "neutral": 1, "positive": 2}
y = df["sentiment_label"].map(label_map).values

print("Unique sentiment labels:", set(df["sentiment_label"]))

# -------------------------------
# 4. Tokenizer + extend vocab
# -------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

emoji_list = df["emoji_clean"].unique().tolist()
tokenizer.add_tokens(emoji_list)

print("Tokenizer vocab size after extension:", len(tokenizer))

# -------------------------------
# 5. Train/Test split
# -------------------------------
X = df["emoji_clean"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

encoded_train = tokenizer(X_train, padding=True, truncation=True, max_length=16, return_tensors="pt")
encoded_test  = tokenizer(X_test, padding=True, truncation=True, max_length=16, return_tensors="pt")

print("\nTrain/Test split complete.")
print("Train size:", encoded_train['input_ids'].shape[0], "Test size:", encoded_test['input_ids'].shape[0])

# -------------------------------
# 6. Save outputs to .pt files
# -------------------------------
save_dir = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_emoji"
os.makedirs(save_dir, exist_ok=True)

torch.save(encoded_train, os.path.join(save_dir, "emoji_train_tokens.pt"))
torch.save(encoded_test, os.path.join(save_dir, "emoji_test_tokens.pt"))
torch.save(torch.tensor(y_train), os.path.join(save_dir, "emoji_train_labels.pt"))
torch.save(torch.tensor(y_test), os.path.join(save_dir, "emoji_test_labels.pt"))

print("\n✅ Preprocessed emoji outputs saved to:", save_dir)

# -------------------------------
# 7. Save human-readable CSV
# -------------------------------
csv_path = os.path.join(save_dir, "emoji_dataset_info.csv")

df_out = pd.DataFrame({
    "emoji": df["Emoji"],
    "emoji_clean": df["emoji_clean"],
    "sentiment_label": df["sentiment_label"],
    "label_numeric": df["sentiment_label"].map(label_map)
})

df_out.to_csv(csv_path, index=False, encoding="utf-8")

print("\n✅ Human-readable CSV saved to:", csv_path)

# -------------------------------
# 8. Save tokenizer for reuse in extraction
# -------------------------------
tokenizer_dir = os.path.join(save_dir, "tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)

tokenizer.save_pretrained(tokenizer_dir)

print("\n✅ Tokenizer with emoji vocab saved for extraction.")

# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Emoji preprocessing pipeline complete!")
print("Dataset shape:", df.shape)
