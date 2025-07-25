import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the compressed CSV
df = pd.read_csv("compressed_data_25mb.csv.gz")

# Drop missing values and convert label to integer
df.dropna(subset=["text", "label"], inplace=True)
df['label'] = df['label'].astype(int)

# Vectorize the text column
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, df['label'])

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved!")
