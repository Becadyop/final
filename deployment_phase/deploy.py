# deploy.py
# -----------------------------------
# Deployment pipeline: load comment, preprocess, extract features, fuse, predict, save output

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sqlalchemy import create_engine

# -------------------------------
# 1. Define model (same as training)
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

# -------------------------------
# 2. Load trained model
# -------------------------------
input_dim = 768   # must match fusion embedding size
hidden_dim = 128
num_classes = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier(input_dim, hidden_dim, num_classes).to(device)
model.load_state_dict(torch.load("sentiment_model_best.pt", map_location=device))
model.eval()

# -------------------------------
# 3. Connect to database and fetch comments
# -------------------------------
# Example: using SQLite, but replace with your DB connection string
engine = create_engine("sqlite:///products.db")
df_comments = pd.read_sql("SELECT id, comment_text FROM product_comments", engine)

# -------------------------------
# 4. Preprocessing (simplified)
# -------------------------------
def preprocess(text):
    # Lowercase, strip, basic cleaning
    return text.lower().strip()

df_comments["clean_text"] = df_comments["comment_text"].apply(preprocess)

# -------------------------------
# 5. Feature extraction + fusion
# -------------------------------
# Assume you already have a function that converts text to fused embeddings
# (e.g., BERT embeddings + image features if available)
def extract_fusion_embedding(text):
    # Placeholder: load your preprocessing + fusion pipeline here
    # For demo, return a random tensor of size (768,)
    return torch.rand(768)

embeddings = torch.stack([extract_fusion_embedding(t) for t in df_comments["clean_text"]])
embeddings = embeddings.to(device)

# -------------------------------
# 6. Prediction
# -------------------------------
with torch.no_grad():
    outputs = model(embeddings)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# Map predictions to labels
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df_comments["sentiment"] = [label_map[p] for p in preds]

# -------------------------------
# 7. Save results back to database
# -------------------------------
df_comments[["id", "sentiment"]].to_sql("product_sentiments", engine, if_exists="replace", index=False)

print("✅ Sentiment predictions saved to product_sentiments table")
