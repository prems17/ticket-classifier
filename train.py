# train.py - trains the text classifier and saves it to model.joblib

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data (a table with two columns: text, category)
data = pd.read_csv("data/tickets.csv")
print("Total tickets:", len(data))
print(data["category"].value_counts(), "\n")

# 2. Split: 75% to teach the model, 25% hidden to test it
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["category"],
    test_size=0.25, random_state=42, stratify=data["category"]
)

# 3. Build the model: TF-IDF turns text into numbers,
#    Logistic Regression learns which numbers mean which category
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, C=10)),
])

# 4. Train
model.fit(X_train, y_train)

# 5. Test on the tickets the model has never seen
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}  ({accuracy * 100:.0f}%)\n")
print(classification_report(y_test, predictions))

# 6. Save the trained model so the web app can reuse it
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")
