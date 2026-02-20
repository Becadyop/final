import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Define main folder
main_folder = r"C:\Users\user\Desktop\Projects\EchoFeeling\data\sticker"

# 2. Collect counts per category
counts = {}
for subfolder in os.listdir(main_folder):
    path = os.path.join(main_folder, subfolder)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(".png")])
        if count > 0:  # skip empty folders
            counts[subfolder] = count

# 3. Convert to DataFrame for easy handling
df = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])

# 4. Sort categories by count
df = df.sort_values("Count", ascending=False)

# 5. Print summary
print("Sticker dataset distribution:")
print(df)
print(f"\nTotal PNG images: {df['Count'].sum()}")

# 6. Plot distribution
# Use a color palette instead of fixed mapping
palette = sns.color_palette("Set2", len(df))

ax = df.plot(kind="bar", x="Category", y="Count", legend=False, color=palette)
plt.title("Sticker Dataset Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")

# 7. Save plot
plt.savefig("sticker_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
