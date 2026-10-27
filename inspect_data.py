import os
import pandas as pd

file_path = os.path.join("data", "phiusiil_raw.csv")

print("Loading dataset from local disk...")
# Load only the URL and target label to keep it fast
df = pd.read_csv(file_path, usecols=["URL", "label"])

print("\n--- Total Records ---")
print(f"Total rows: {len(df):,}")

print("\n--- Class Distribution ---")
label_counts = df["label"].value_counts()
print(label_counts)

print("\n--- Percentage Distribution ---")
label_percentage = df["label"].value_counts(normalize=True) * 100
print(f"Legitimate (1): {label_percentage.get(1, 0):.2f}%")
print(f"Phishing   (0): {label_percentage.get(0, 0):.2f}%")

print("\n--- Sample Legitimate URLs (label = 1) ---")
print(df[df["label"] == 1]["URL"].head(3).to_list())

print("\n--- Sample Phishing URLs (label = 0) ---")
print(df[df["label"] == 0]["URL"].head(3).to_list())