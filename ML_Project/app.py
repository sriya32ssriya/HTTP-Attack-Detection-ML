from flask import Flask, request, render_template_string
import joblib
import re

app = Flask(__name__)

# =============================
# Load trained model & vectorizer
# =============================
model = joblib.load("model/nb_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# =============================
# HTTP Input Extraction
# =============================
def extract_from_http_input(text):
    """
    Handles:
    - Normal URLs
    - Full HTTP requests
    - Plain text
    """
    text = str(text)
    lines = text.splitlines()
    extracted = []

    # First line: URL or request line
    if len(lines) > 0:
        extracted.append(lines[0])

    # Body (if present)
    if "" in lines:
        body_index = lines.index("") + 1
        extracted.append(" ".join(lines[body_index:]))

    return " ".join(extracted)


# =============================
# Cleaning function (same as training)
# =============================
def clean_payload(text):
    text = extract_from_http_input(text)
    text = text.lower()
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^\w\s<>/=]', ' ', text)
    return text


# =============================
# HTML UI (Simple & Clean)
# =============================
HTML = """
<h2>HTTP Request / URL Attack Detection</h2>

<form method="post">
<textarea name="input_data" rows="7" cols="90"
placeholder="Paste a URL or full HTTP request here"></textarea><br><br>
<button type="submit">Analyze</button>
</form>

{% if result %}
<h3 style="color:green;">Detected Attack Type: {{ result }}</h3>
{% endif %}

<p><i>Model: Naive Bayes with TF-IDF features</i></p>
"""


# =============================
# Flask Route
# =============================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        user_input = request.form["input_data"]

        cleaned = clean_payload(user_input)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

    return render_template_string(HTML, result=result)


# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=True)