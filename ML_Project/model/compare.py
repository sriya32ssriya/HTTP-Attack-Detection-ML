import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report


# =============================
# 1. Load prepared datasets
# =============================
train_df = pd.read_csv("../dataset/train_final.csv")
test_df  = pd.read_csv("../dataset/test_final.csv")

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# =============================
# 2. Text cleaning function
# =============================
def clean_payload(text):
    text = str(text).lower()
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^\w\s<>/=]', ' ', text)
    return text


# Apply cleaning
train_df["cleaned_payload"] = train_df["payload"].apply(clean_payload)
test_df["cleaned_payload"]  = test_df["payload"].apply(clean_payload)

print("\nSample cleaned payloads:")
print(train_df[["payload", "cleaned_payload"]].head())


# =============================
# 3. TF-IDF Vectorization
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
# 4. Logistic Regression
# =============================
print("\n==============================")
print(" Logistic Regression ")
print("==============================")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))


# =============================
# 5. Naive Bayes
# =============================
print("\n==============================")
print(" Naive Bayes ")
print("==============================")

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


# =============================
# 6. Linear SVM
# =============================
print("\n==============================")
print(" Linear SVM ")
print("==============================")

svm_model = LinearSVC(class_weight="balanced")
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


print("\n Model comparison completed.")