import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# -----------------------------
# 1. Paths
# -----------------------------
preprocessed_path = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_sticker"
extraction_output_path = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extraction_sticker_output"
os.makedirs(extraction_output_path, exist_ok=True)

# -----------------------------
# 2. Load preprocessed data (.pt files)
# -----------------------------
X_train = torch.load(os.path.join(preprocessed_path, "stickers_X_train.pt"), weights_only=False)
y_train = torch.load(os.path.join(preprocessed_path, "stickers_y_train.pt"), weights_only=False)
X_test  = torch.load(os.path.join(preprocessed_path, "stickers_X_test.pt"), weights_only=False)
y_test  = torch.load(os.path.join(preprocessed_path, "stickers_y_test.pt"), weights_only=False)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------------
# 3. Define CNN model
# -----------------------------
class StickerCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # Input 128x128 → after two poolings → 32x32
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        embeddings = torch.relu(self.fc1(x))
        logits = self.fc2(embeddings)
        return torch.softmax(logits, dim=1), embeddings

# -----------------------------
# 4. Run extraction
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StickerCNN().to(device)
model.eval()

dataset = TensorDataset(X_test, y_test)
loader = DataLoader(dataset, batch_size=8)

all_preds, all_embeds, all_labels = [], [], []
with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        preds, embeds = model(imgs)
        all_preds.append(preds.cpu())
        all_embeds.append(embeds.cpu())
        all_labels.append(lbls.cpu())

preds = torch.cat(all_preds)
embeds = torch.cat(all_embeds)
labels = torch.cat(all_labels)

print("Embeddings shape:", embeds.shape)

# -----------------------------
# 5. Save outputs
# -----------------------------
torch.save({
    "embeddings": embeds,
    "predictions": preds,
    "labels": labels
}, os.path.join(extraction_output_path, "stickers_extraction.pt"))

df = pd.DataFrame(preds.numpy(), columns=["negative_prob", "neutral_prob", "positive_prob"])
df["true_label"] = labels.numpy()
# Add embeddings as extra columns
emb_cols = [f"emb_{i}" for i in range(embeds.shape[1])]
df_emb = pd.DataFrame(embeds.numpy(), columns=emb_cols)
df_out = pd.concat([df, df_emb], axis=1)

df_out.to_csv(os.path.join(extraction_output_path, "stickers_extraction.csv"), index=False)

print("✅ Sticker extraction complete. Outputs saved to:", extraction_output_path)
