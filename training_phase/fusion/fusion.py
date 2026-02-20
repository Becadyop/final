# fusion_torch.py
# -----------------------------------
# Fusion of text, emoji, and sticker features using PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

# -------------------------------
# 1. Load extracted features (.pt files)
# -------------------------------
# Text: separate files for embeddings and labels
text_train_embeds = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extracted_output_text\text_train_embeddings.pt"
)
text_train_labels = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extracted_output_text\text_train_labels.pt"
)
text_test_embeds = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extracted_output_text\text_test_embeddings.pt"
)
text_test_labels = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extracted_output_text\text_test_labels.pt"
)

# Emoji: single dict file
emoji_data = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extraction_emoji_output\emoji_extraction.pt"
)

# Sticker: single dict file
sticker_data = torch.load(
    r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\extraction_sticker_output\stickers_extraction.pt"
)

# -------------------------------
# 2. Extract embeddings and labels
# -------------------------------
emoji_train_embeds = emoji_data["train_embeddings"]
emoji_train_labels = emoji_data["train_labels"]

sticker_train_embeds = sticker_data["embeddings"]
sticker_train_labels = sticker_data["labels"]

# -------------------------------
# 2b. Pad emoji/sticker to match text samples
# -------------------------------
N = text_train_embeds.shape[0]

def pad_to_match(X, N):
    if X.shape[0] < N:
        pad = torch.zeros(N - X.shape[0], X.shape[1])
        return torch.cat([X, pad], dim=0)
    elif X.shape[0] > N:
        return X[:N]  # truncate if larger
    else:
        return X

emoji_train_embeds   = pad_to_match(emoji_train_embeds, N)
sticker_train_embeds = pad_to_match(sticker_train_embeds, N)

print("Aligned shapes:", text_train_embeds.shape,
      emoji_train_embeds.shape,
      sticker_train_embeds.shape)

# -------------------------------
# 3. Fusion Module
# -------------------------------
class FusionLayer(nn.Module):
    def __init__(self, text_dim, emoji_dim, sticker_dim, target_dim):
        super().__init__()
        self.proj_text    = nn.Linear(text_dim, target_dim)
        self.proj_emoji   = nn.Linear(emoji_dim, target_dim)
        self.proj_sticker = nn.Linear(sticker_dim, target_dim)
        self.gate_emoji   = nn.Linear(target_dim, 1)
        self.gate_sticker = nn.Linear(target_dim, 1)

    def forward(self, text, emoji, sticker):
        text_proj    = F.normalize(self.proj_text(text), p=2, dim=1)
        emoji_proj   = F.normalize(self.proj_emoji(emoji), p=2, dim=1)
        sticker_proj = F.normalize(self.proj_sticker(sticker), p=2, dim=1)

        g_e = torch.sigmoid(self.gate_emoji(text_proj))
        g_s = torch.sigmoid(self.gate_sticker(text_proj))

        emoji_fused   = g_e * emoji_proj
        sticker_fused = g_s * sticker_proj

        return text_proj + emoji_fused + sticker_fused

# -------------------------------
# 4. Run fusion (train set example)
# -------------------------------
target_dim = max(text_train_embeds.shape[1],
                 emoji_train_embeds.shape[1],
                 sticker_train_embeds.shape[1])

fusion_layer = FusionLayer(text_train_embeds.shape[1],
                           emoji_train_embeds.shape[1],
                           sticker_train_embeds.shape[1],
                           target_dim)

fusion_train = fusion_layer(text_train_embeds,
                            emoji_train_embeds,
                            sticker_train_embeds)

print("Final fusion shape (train):", fusion_train.shape)

# -------------------------------
# 5. Save outputs
# -------------------------------
save_dir = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\extraction\fusion_output"
os.makedirs(save_dir, exist_ok=True)

torch.save({
    "fusion_train_embeddings": fusion_train,
    "fusion_train_labels": text_train_labels
}, os.path.join(save_dir, "fusion_train.pt"))

labels = text_train_labels
print("Unique labels in fusion:", torch.unique(labels))
print("Counts per class:", torch.bincount(labels))


# Save CSV for inspection
df_out = pd.DataFrame(fusion_train.detach().numpy())
df_out["label"] = text_train_labels.numpy()
df_out.to_csv(os.path.join(save_dir, "fusion_train.csv"), index=False)

print("Unique labels in fusion:", torch.unique(text_train_labels))
print("Counts per class:", torch.bincount(text_train_labels))


print("\n✅ Fusion outputs saved in:", save_dir)
