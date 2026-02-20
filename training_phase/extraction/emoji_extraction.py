import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel   # ✅ Added DistilBertModel

print("✅ Emoji extraction script started", flush=True)

# -------------------------------
# 1. Load tokenizer (extended with emojis during preprocessing)
# ------------------------------
tokenizer_path = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_emoji\tokenizer"
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


# -------------------------------
# 2. Load DistilBERT + resize embeddings
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert.resize_token_embeddings(len(tokenizer))
bert.eval()

# -------------------------------
# 3. Load preprocessed train/test data (weights_only=False)
# -------------------------------
train_tokens = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_emoji\emoji_train_tokens.pt",
    weights_only=False
)
train_labels = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_emoji\emoji_train_labels.pt",
    weights_only=False
)

test_tokens = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_emoji\emoji_test_tokens.pt",
    weights_only=False
)
test_labels = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_emoji\emoji_test_labels.pt",
    weights_only=False
)

# -------------------------------
# 4. Define classifier head
# -------------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=3):
        super().__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] vector
        logits = self.fc(cls_embedding)
        return torch.softmax(logits, dim=1), cls_embedding

classifier = SentimentClassifier(bert).to(device)
classifier.eval()

# -------------------------------
# 5. Run extraction (batched)
# -------------------------------
def run_extraction(encoded_batch, labels, batch_size=32):
    dataset = TensorDataset(encoded_batch["input_ids"], encoded_batch["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds, all_embeddings, all_labels = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, lbls in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            preds, embeddings = classifier(input_ids, attention_mask)
            all_preds.append(preds.cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(lbls.cpu())

    return torch.cat(all_preds), torch.cat(all_embeddings), torch.cat(all_labels)

train_preds, train_embeddings, train_labels = run_extraction(train_tokens, train_labels)
test_preds, test_embeddings, test_labels   = run_extraction(test_tokens, test_labels)

print("Train embeddings shape:", train_embeddings.shape)
print("Test embeddings shape:", test_embeddings.shape)

# -------------------------------
# 6. Save extracted outputs
# -------------------------------
save_dir = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extraction_emoji_output"
os.makedirs(save_dir, exist_ok=True)

torch.save({
    "train_embeddings": train_embeddings,
    "train_predictions": train_preds,
    "train_labels": train_labels,
    "test_embeddings": test_embeddings,
    "test_predictions": test_preds,
    "test_labels": test_labels
}, os.path.join(save_dir, "emoji_extraction.pt"))

# -------------------------------
# 7. Save human-readable CSV
# -------------------------------
df_train = pd.DataFrame(train_embeddings.numpy())
df_train["label"] = train_labels.numpy()
df_train["predicted"] = train_preds.argmax(dim=1).numpy()

df_test = pd.DataFrame(test_embeddings.numpy())
df_test["label"] = test_labels.numpy()
df_test["predicted"] = test_preds.argmax(dim=1).numpy()

df_out = pd.concat([df_train, df_test], axis=0)
df_out.to_csv(os.path.join(save_dir, "emoji_extraction_info.csv"), index=False)

print("\n✅ Emoji extraction complete. Outputs saved to:", save_dir)
