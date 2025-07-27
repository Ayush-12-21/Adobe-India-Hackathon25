# train_classifier.py
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
with open("training.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(texts)

# Train classifier
clf = LogisticRegression()
clf.fit(X, labels)

# Save model and vectorizer
with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Classifier trained and saved.")
