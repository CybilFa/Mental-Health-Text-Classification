from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords

app = Flask(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../Model Training and Testing")

# --- Stopwords ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --- Load model and vectorizer ---
print("Loading model and vectorizer...")
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'mental_health_model.pkl'))
print("Model loaded successfully!")

# --- Clean text function ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text.strip()

# --- Label mapping ---
label_map = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar Disorder",
    3: "Personality Disorder",
    4: "Anxiety"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "Please enter text for prediction."}), 400

        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        disorder = label_map.get(prediction, "Unknown")

        return jsonify({"prediction": disorder})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
