# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset (must be in the same folder)
df = pd.read_csv("compressed_data_25mb.csv.gz")

# Clean the data
df.dropna(subset=["text", "label"], inplace=True)
df["label"] = df["label"].astype(int)

# Create vectorizer and transform text
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])

# Create and train model
model = MultinomialNB()
model.fit(X, df["label"])

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved!")
