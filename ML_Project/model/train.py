import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# =============================
# 1. Load prepared datasets
# =============================
train_df = pd.read_csv("../dataset/train_final.csv")
test_df  = pd.read_csv("../dataset/test_final.csv")

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# =============================
# 2. HTTP Input Extraction
# =============================
def extract_from_http_input(text):
    """
    Handles:
    - Full URLs
    - Full HTTP requests
    - Plain text
    """
    text = str(text)
    lines = text.splitlines()
    extracted = []

    # Request line or URL
    if len(lines) > 0:
        extracted.append(lines[0])

    # Body (if present)
    if "" in lines:
        body_index = lines.index("") + 1
        extracted.append(" ".join(lines[body_index:]))

    return " ".join(extracted)


# =============================
# 3. Text Cleaning Function
# =============================
def clean_payload(text):
    text = extract_from_http_input(text)
    text = text.lower()
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^\w\s<>/=]', ' ', text)
    return text


# Apply cleaning
train_df["cleaned_payload"] = train_df["payload"].apply(clean_payload)
test_df["cleaned_payload"]  = test_df["payload"].apply(clean_payload)

print("\nSample cleaned payloads:")
print(train_df[["payload", "cleaned_payload"]].head())


# =============================
# 4. TF-IDF Vectorization
# =============================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_df["cleaned_payload"])
X_test  = vectorizer.transform(test_df["cleaned_payload"])

y_train = train_df["label"]
y_test  = test_df["label"]

print("\nTF-IDF train shape:", X_train.shape)
print("TF-IDF test shape:", X_test.shape)


# =============================
# 5. Train Naive Bayes Model
# =============================
print("\n==============================")
print(" Naive Bayes Model ")
print("==============================")

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)


# =============================
# 6. Evaluation Metrics (Terminal)
# =============================
accuracy  = accuracy_score(y_test, nb_pred)
precision = precision_score(y_test, nb_pred, average="weighted")
recall    = recall_score(y_test, nb_pred, average="weighted")
f1        = f1_score(y_test, nb_pred, average="weighted")

print("\n==============================")
print(" MODEL EVALUATION METRICS ")
print("==============================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n------------------------------")
print(" Classification Report ")
print("------------------------------")
print(classification_report(y_test, nb_pred))


# =============================
# 7. Confusion Matrix (GRAPH)
# =============================
cm = confusion_matrix(y_test, nb_pred, labels=["SQLi", "XSS"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["SQLi", "XSS"],
    yticklabels=["SQLi", "XSS"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.show()


# =============================
# 8. Metrics Bar Graph
# =============================
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(7, 5))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Naive Bayes Performance Metrics")
plt.tight_layout()
plt.show()


print("\n Evaluation completed successfully.")