"""
app.py - Flask Backend API

LABEL CONVENTION (must match clean.py & model.py):
  0 = FAKE (Computer Generated)
  1 = REAL (Original)

model.predict()       → 0 = FAKE, 1 = REAL
model.predict_proba() → [P(FAKE), P(REAL)]  (because classes_ = [0, 1])
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import numpy as np

app = Flask(__name__)
CORS(app)

# ========== LOAD MODEL ==========
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("❌ model.pkl or vectorizer.pkl not found!")
        print("   Run:  python clean.py  →  python model.py")
        return None, None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    # ═══ VERIFY CLASS ORDER ═══
    print(f"✅ Model loaded. classes_ = {model.classes_}")
    print(f"   Index 0 = class {model.classes_[0]} → {'FAKE' if model.classes_[0] == 0 else 'REAL'}")
    print(f"   Index 1 = class {model.classes_[1]} → {'FAKE' if model.classes_[1] == 0 else 'REAL'}")

    return model, vectorizer

model, vectorizer = load_model()

# ========== DETERMINE CLASS INDICES ==========
# This is the KEY FIX — we dynamically find which index is FAKE and which is REAL
# regardless of how model.classes_ is ordered
FAKE_INDEX = None
REAL_INDEX = None

if model is not None:
    classes = model.classes_.tolist()
    FAKE_INDEX = classes.index(0)   # label 0 = FAKE
    REAL_INDEX = classes.index(1)   # label 1 = REAL
    print(f"   FAKE probability index: {FAKE_INDEX}")
    print(f"   REAL probability index: {REAL_INDEX}")

# ========== TEXT CLEANING ==========
def clean_input(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== API ROUTES ==========

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Fake Review Detection API',
        'labels': {'0': 'FAKE', '1': 'REAL'},
        'endpoints': {
            '/predict': 'POST — Send {"review": "your text"} to get prediction'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Please send JSON with a "review" field.'}), 400

    review_text = data['review'].strip()
    if len(review_text) < 10:
        return jsonify({'error': 'Review too short. Enter at least 10 characters.'}), 400

    # Clean → Vectorize → Predict
    cleaned = clean_input(review_text)
    text_vector = vectorizer.transform([cleaned])

    prediction = model.predict(text_vector)[0]          # 0=FAKE or 1=REAL
    probabilities = model.predict_proba(text_vector)[0]  # [P(class_0), P(class_1)]

    # ═══ Use correct indices ═══
    fake_prob = float(probabilities[FAKE_INDEX])
    real_prob = float(probabilities[REAL_INDEX])

    is_fake = (prediction == 0)

    result = {
        'prediction': 'FAKE' if is_fake else 'REAL',
        'label': int(prediction),
        'confidence': round(max(fake_prob, real_prob) * 100, 2),
        'probabilities': {
            'fake': round(fake_prob * 100, 2),
            'real': round(real_prob * 100, 2)
        },
        'review_snippet': review_text[:100] + ('...' if len(review_text) > 100 else '')
    }

    # Debug log
    print(f"  Review: \"{review_text[:60]}...\"")
    print(f"  Prediction: {result['prediction']} | Confidence: {result['confidence']}%")
    print(f"  Probabilities: Fake={result['probabilities']['fake']}%, Real={result['probabilities']['real']}%")

    return jsonify(result)

# ========== RUN ==========
if __name__ == '__main__':
    print(f"\n{'='*55}")
    print(f"  FAKE REVIEW DETECTION API")
    print(f"  URL: http://127.0.0.1:5000")
    print(f"  Labels: 0=FAKE, 1=REAL")
    print(f"{'='*55}\n")
    app.run(debug=True, port=5000)
