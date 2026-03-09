import pandas as pd

# -----------------------------
# LOAD DATASETS
# -----------------------------
sqli_train = pd.read_csv("../dataset/SQLI/SQLI_COMBINED_DATASET_Train.csv")
sqli_test  = pd.read_csv("../dataset/SQLI/SQLI_TEST_Dataset.csv")

xss_train  = pd.read_csv("../dataset/XSS/xss_dataset_train.csv")
xss_test   = pd.read_csv("../dataset/XSS/xss_dataset_test.csv")

# -----------------------------
# KEEP ONLY PAYLOAD COLUMN
# -----------------------------
sqli_train = sqli_train[["payload"]]
sqli_test  = sqli_test[["payload"]]

xss_train  = xss_train[["payload"]]
xss_test   = xss_test[["payload"]]

# -----------------------------
# ADD CLEAN LABELS
# -----------------------------
sqli_train["label"] = "SQLi"
sqli_test["label"]  = "SQLi"

xss_train["label"]  = "XSS"
xss_test["label"]   = "XSS"

# -----------------------------
# MERGE DATASETS
# -----------------------------
train_df = pd.concat([sqli_train, xss_train], ignore_index=True)
test_df  = pd.concat([sqli_test, xss_test], ignore_index=True)

# -----------------------------
# SAVE FINAL DATASETS
# -----------------------------
train_df.to_csv("../dataset/train_final.csv", index=False)
test_df.to_csv("../dataset/test_final.csv", index=False)

print("✅ Final datasets created successfully!")
print("Training samples:")
print(train_df["label"].value_counts())
print("\nTesting samples:")
print(test_df["label"].value_counts())
