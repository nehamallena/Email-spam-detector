import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv("compressed_data_25mb.csv.gz")
df.dropna(subset=["text", "label"], inplace=True)
df['label'] = df['label'].astype(int)

# Vectorize and train
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved!")
