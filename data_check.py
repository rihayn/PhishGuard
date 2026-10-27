import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Define where to save the file
output_path = os.path.join("data", "phiusiil_raw.csv")

if os.path.exists(output_path):
    print(f"Dataset already exists locally at: {output_path}")
else:
    print("Fetching dataset from UCI...")
    phishing_data = fetch_ucirepo(id=967)

    X = phishing_data.data.features
    y = phishing_data.data.targets

    print("Combining features and labels...")
    # Add the target column 'label' to our dataframe
    full_df = pd.concat([X, y], axis=1)

    print(f"Saving to {output_path} (this takes just a few seconds)...")
    full_df.to_csv(output_path, index=False)
    print("Done! Dataset saved locally.")

# Quick verification: read the first 2 rows from the local file
df_local = pd.read_csv(output_path, nrows=2)
print("\nLocal file verified! Columns present in the file:")
print(list(df_local.columns))