# train.py
# -----------------------------------
# Training sentiment classifier on fused embeddings with inverse frequency weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import datetime

# 1. Load fused data
data = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\fusion\fusion_output\fusion_train.pt"
)
features, labels = data["fusion_train_embeddings"], data["fusion_train_labels"]

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("Unique labels:", torch.unique(labels))
print("Counts per class:", torch.bincount(labels))

# -------------------------------
# 2. Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features, labels = features.to(device), labels.to(device)

# -------------------------------
# 3. Train/Val/Test split
# -------------------------------
dataset = TensorDataset(features, labels)
train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32)
test_loader  = DataLoader(test_set, batch_size=32)

# -------------------------------
# 4. Classifier model
# -------------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out

input_dim = features.shape[1]
hidden_dim = 128
num_classes = len(torch.unique(labels))
model = SentimentClassifier(input_dim, hidden_dim, num_classes).to(device)

# -------------------------------
# 5. Loss & Optimizer (inverse frequency weights)
# -------------------------------
class_counts = torch.bincount(labels)
class_weights = (1.0 / class_counts.float()).to(device)
print("Class weights (inverse frequency):", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------
# 6. Training loop with validation + early stopping
# -------------------------------
epochs = 50
best_val_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            val_loss += criterion(outputs, y_val).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"sentiment_model_best.pt")
        print("✅ Validation improved, model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered (no improvement).")
            break

# -------------------------------
# 7. Evaluation on test set (with full report)
# -------------------------------
labels_list = [0, 1, 2]
target_names = ["negative", "neutral", "positive"]

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

f1_weighted = f1_score(all_labels, all_preds, average="weighted")
f1_macro = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
report = classification_report(all_labels, all_preds,
                               labels=labels_list,
                               target_names=target_names,
                               output_dict=True)

print("\n✅ Evaluation Results")
print("Weighted F1 Score:", f1_weighted)
print("Macro F1 Score:", f1_macro)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(all_labels, all_preds,
                            labels=labels_list,
                            target_names=target_names))

# -------------------------------
# 8. Save trained model & results with timestamp
# -------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f"sentiment_model_weights_{timestamp}.pt")

results = {
    "F1_weighted": [f1_weighted],
    "F1_macro": [f1_macro],
    "Confusion_matrix": [cm.tolist()],
    "Per_class_metrics": [report],
    "Weights_used": [class_weights.tolist()]
}
df_results = pd.DataFrame(results)
df_results.to_csv(f"evaluation_results_weights_{timestamp}.csv", index=False)

print(f"\n✅ Results saved as evaluation_results_weights_{timestamp}.csv")

# ------------------------------- 
# 9. Save evaluation results to CSV (latest run)
# -------------------------------
results = {
    "F1_weighted": [f1_weighted],
    "F1_macro": [f1_macro],
    "Confusion_matrix": [cm.tolist()]
}
df_results = pd.DataFrame(results)
df_results.to_csv("evaluation_results.csv", index=False)
print("\n✅ Evaluation results saved to evaluation_results.csv")
