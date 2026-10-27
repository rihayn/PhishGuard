import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.features import extract_features_from_url

# 1. Load data from local CSV
csv_path = os.path.join("data", "phiusiil_raw.csv")
print("[1/5] Loading data from disk...")
df = pd.read_csv(csv_path, usecols=["URL", "label"])

# 2. Take a balanced sample (10,000 legitimate + 10,000 phishing)
print("[2/5] Sampling 20,000 balanced records...")
legit = df[df["label"] == 1].sample(n=10000, random_state=42)
phish = df[df["label"] == 0].sample(n=10000, random_state=42)
sample_df = pd.concat([legit, phish]).reset_index(drop=True)

# 3. Extract features using our custom function from src/features.py
print("[3/5] Extracting 7 lexical features for all sample URLs...")
feature_rows = [extract_features_from_url(u) for u in sample_df["URL"]]
X = pd.DataFrame(feature_rows)
y = sample_df["label"]

# 4. Split into Train (80%) and Test (20%) sets
print("[4/5] Splitting into Train (80%) and Test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train the Logistic Regression model
print("[5/5] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on unseen test URLs
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 40)
print(f"Model Accuracy on Unseen URLs: {accuracy * 100:.2f}%")
print("=" * 40)
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred, target_names=["Phishing (0)", "Legitimate (1)"]))