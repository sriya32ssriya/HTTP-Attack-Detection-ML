import pandas as pd

# Load Kaggle dataset
df = pd.read_csv(
    r"C:\Users\Pavan\Downloads\archive\Modified_SQL_Dataset.csv"
)

# Convert to required format
df_new = pd.DataFrame()
df_new["payload"] = df["Query"]
df_new["length"] = df_new["payload"].apply(len)
df_new["attack_type"] = "sqli"
df_new["label"] = df["Label"].apply(lambda x: "anom" if x == 1 else "norm")

# Save locally (Windows-friendly)
df_new.to_csv("kaggle_sqli_converted.csv", index=False)

print("✅ Converted dataset saved as kaggle_sqli_converted.csv")
