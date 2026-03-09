import joblib
import re

# Load saved model and vectorizer
model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_payload(text):
    text = str(text).lower()
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^\w\s<>/=]', ' ', text)
    return text

def predict_attack(payload):
    cleaned = clean_payload(payload)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

# Test samples
tests = [
    "1 OR 1=1",
    "<script>alert(1)</script>",
    "/home?id=5"
]

for t in tests:
    print(t, "=>", predict_attack(t))
