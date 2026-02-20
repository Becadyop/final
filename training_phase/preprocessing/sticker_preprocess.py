import os
import torch
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import pandas as pd

# Paths to your dataset folders
base_path = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\data\sticker"
categories = ["negative", "neutral", "positive"]

# Output folder for preprocessed files
output_path = r"C:\Users\user\Desktop\Projects\EchoFeeling\training_phase\preprocessing\preprocessed_output_sticker"
os.makedirs(output_path, exist_ok=True)

# Parameters
img_size = (128, 128)  # resize size for custom CNN

data = []
labels = []
filenames = []

# Load, resize, normalize, and label encode
for label, folder in enumerate(categories):
    folder_path = os.path.join(base_path, folder)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = load_img(img_path, target_size=img_size)   # resize
        img_array = img_to_array(img) / 255.0            # normalize to [0,1]
        # Convert to channel-first format for PyTorch: (H,W,C) -> (C,H,W)
        img_array = np.transpose(img_array, (2, 0, 1))
        data.append(img_array)
        labels.append(label)
        filenames.append(file)

# Convert to numpy arrays
X = np.array(data, dtype=np.float32)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save arrays as PyTorch tensors
torch.save(torch.tensor(X_train), os.path.join(output_path, "stickers_X_train.pt"))
torch.save(torch.tensor(y_train), os.path.join(output_path, "stickers_y_train.pt"))
torch.save(torch.tensor(X_test), os.path.join(output_path, "stickers_X_test.pt"))
torch.save(torch.tensor(y_test), os.path.join(output_path, "stickers_y_test.pt"))


# Save human-readable CSV (mapping file names to labels)
df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})
df["label_name"] = df["label"].map({0:"negative", 1:"neutral", 2:"positive"})
df.to_csv(os.path.join(output_path, "stickers_dataset_info.csv"), index=False)

print("✅ Preprocessing complete. Files saved in:")
print(output_path)
