# text_extraction_pipeline.py
# -----------------------------
# Extract sentence-level embeddings from preprocessed Amazon reviews
# Uses DistilBERT for efficiency (low RAM/CPU)
# Saves outputs to extracted_output_text folder

import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel

print("✅ Text extraction script started", flush=True)

# -------------------------------
# 1. Load preprocessed .pt files
# -------------------------------
preprocessed_dir = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_text"

train_tokens = torch.load(
    os.path.join(preprocessed_dir, "text_train_tokens.pt"),
    weights_only=False
)
test_tokens = torch.load(
    os.path.join(preprocessed_dir, "text_test_tokens.pt"),
    weights_only=False
)
y_train = torch.load(
    os.path.join(preprocessed_dir, "text_train_labels.pt"),
    weights_only=False
)
y_test = torch.load(
    os.path.join(preprocessed_dir, "text_test_labels.pt"),
    weights_only=False
)


print("Loaded preprocessed tensors.")
print("Train size:", train_tokens["input_ids"].shape[0], "Test size:", test_tokens["input_ids"].shape[0])

# -------------------------------
# 2. Initialize DistilBERT model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

print("DistilBERT model loaded on", device)

# -------------------------------
# 3. Generate embeddings (batched)
# -------------------------------
def get_sentence_embeddings(encoded_batch, batch_size=32):
    """Return [CLS] embeddings for each sentence in batch"""
    dataset = TensorDataset(encoded_batch["input_ids"], encoded_batch["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] vector
            all_embeddings.append(cls_embeddings.cpu())  # keep on CPU for saving

    return torch.cat(all_embeddings, dim=0)

train_embeddings = get_sentence_embeddings(train_tokens, batch_size=32)
test_embeddings  = get_sentence_embeddings(test_tokens, batch_size=32)

print("Train embeddings shape:", train_embeddings.shape)
print("Test embeddings shape:", test_embeddings.shape)

# -------------------------------
# 4. Save extracted embeddings
# -------------------------------
save_dir = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extracted_output_text"
os.makedirs(save_dir, exist_ok=True)

torch.save(train_embeddings, os.path.join(save_dir, "text_train_embeddings.pt"))
torch.save(test_embeddings, os.path.join(save_dir, "text_test_embeddings.pt"))
torch.save(y_train, os.path.join(save_dir, "text_train_labels.pt"))
torch.save(y_test, os.path.join(save_dir, "text_test_labels.pt"))


print("\n✅ Sentence embeddings saved to:", save_dir)


# -------------------------------
# 5. Save human-readable CSV
# -------------------------------
import pandas as pd

csv_path = os.path.join(save_dir, "text_embeddings_info.csv")

# Convert embeddings to numpy for saving (optional: you can truncate for readability)
train_np = train_embeddings.numpy()
test_np  = test_embeddings.numpy()

# Create DataFrame with labels and embeddings
df_train = pd.DataFrame(train_np)
df_train["label"] = y_train.numpy()

df_test = pd.DataFrame(test_np)
df_test["label"] = y_test.numpy()

# Concatenate train + test for one CSV
df_out = pd.concat([df_train, df_test], axis=0)

df_out.to_csv(csv_path, index=False, encoding="utf-8")

print("\n✅ Human-readable CSV saved to:", csv_path)


# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Text extraction pipeline complete!")
print("Train embeddings:", train_embeddings.shape, "Test embeddings:", test_embeddings.shape)
