import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading data from data/phiusiil_raw.csv...")
df = pd.read_csv("data/phiusiil_raw.csv")

# 2. Define Features to Remove
# Text/Metadata columns
metadata_cols = ["URL", "Domain", "TLD", "Title", "label"]

# Webpage / HTML / Content-based features (requires downloading page content)
content_cols = [
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef"
]

cols_to_drop = metadata_cols + content_cols

# Keep only columns that exist in the dataframe
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df["label"]

print(f"Training on {X.shape[1]} URL-only features:")
print(list(X.columns))

# 3. Train/Test Split (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train Random Forest
print("\nTraining Random Forest model on URL-only clues...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("          URL-ONLY MODEL RESULTS")
print("=" * 50)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 6. Top 10 URL Clues (Feature Importance)
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n--- Top 10 Most Predictive URL Features ---")
print(importance_df.head(10).to_string(index=False))
print("=" * 50)








import joblib

# Save the trained model and the exact list of feature columns
joblib.dump({"model": rf_model, "features": list(X.columns)}, "phishguard_model.joblib")
print("\n[SUCCESS] Model and feature list saved to 'phishguard_model.joblib'!")