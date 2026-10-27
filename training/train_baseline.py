
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the saved datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df["tweet"]
y_train = train_df["label"]

X_test = test_df["tweet"]
y_test = test_df["label"]

# Step 2: Build the baseline model
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Step 3: Train the model
print("Training baseline model...")
model.fit(X_train, y_train)

# Step 4: Make predictions on unseen test tweets
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["false-negative", "other-neutral", "true-positive"]
))


# Step 6: Save the trained model
joblib.dump(model, "models/tfidf_logistic_regression.joblib")

print("\nModel saved successfully!")
