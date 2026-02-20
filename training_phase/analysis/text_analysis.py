import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("../data/text/amazon_reviews.csv", encoding="utf-8")

# 2. Inspect columns
print("First 5 rows:")
print(df.head())
print("Column names:", df.columns)

# 3. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())

df["cleaned_text"] = df["verified_reviews"].apply(clean_text)   # use verified_reviews column

# 4. Convert rating into sentiment labels (>=4 positive, else negative)
def label_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["rating"].apply(label_sentiment)


# 5. Sentiment distribution
print(df["sentiment"].value_counts())
df["sentiment"].value_counts().plot(kind="bar", color=["green","red"])
plt.title("Sentiment Distribution")
plt.show()

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["sentiment"], test_size=0.2, random_state=42
)

# 7. Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 9. Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

print("Dataset shape:", df.shape)
