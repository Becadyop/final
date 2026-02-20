import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("../data/emoji/Emoji_Sentiment_Data_v1.0.csv", encoding="utf-8")

# 2. Compute sentiment score
df["sentiment_score"] = (df["Positive"] - df["Negative"]) / df["Occurrences"]

# 3. Filter rare emojis (at least 100 occurrences)
df_filtered = df[df["Occurrences"] >= 100]

# 4. Top positive and negative emojis
top_pos = df_filtered.sort_values("sentiment_score", ascending=False).head(10)
top_neg = df_filtered.sort_values("sentiment_score").head(10)

print("Top Positive Emojis:")
print(top_pos[["Emoji", "sentiment_score", "Occurrences"]])

print("\nTop Negative Emojis:")
print(top_neg[["Emoji", "sentiment_score", "Occurrences"]])

# 5. Visualization with proper emoji font
plt.rcParams["font.family"] = "Segoe UI Emoji"  # Windows emoji font

# Positive plot
plt.bar(top_pos["Emoji"], top_pos["sentiment_score"], color="green")
plt.title("Top Positive Emojis")
plt.xlabel("Emoji")
plt.ylabel("Sentiment Score")
plt.xticks(rotation=0, fontsize=14)  # keep emojis upright
plt.show()

# Negative plot
plt.bar(top_neg["Emoji"], top_neg["sentiment_score"], color="red")
plt.title("Top Negative Emojis")
plt.xlabel("Emoji")
plt.ylabel("Sentiment Score")
plt.xticks(rotation=0, fontsize=14)
plt.show()


