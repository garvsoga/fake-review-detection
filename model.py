"""
model.py - ML Model Training Script

IMPORTANT LABEL CONVENTION:
  0 = FAKE (Computer Generated)
  1 = REAL (Original)

predict_proba() returns: [P(class_0), P(class_1)] = [P(FAKE), P(REAL)]
predict() returns: 0 = FAKE, 1 = REAL
"""

import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(data_path='reviews_clean.csv'):
    """Train and save the fake review detection model."""

    print("=" * 55)
    print("  FAKE REVIEW DETECTION — MODEL TRAINING")
    print("=" * 55)

    if not os.path.exists(data_path):
        print(f"\n❌ '{data_path}' not found!")
        print("   Run clean.py first:  python clean.py")
        return

    # ========== 1. LOAD DATA ==========
    print(f"\n[1/7] Loading cleaned dataset...")
    df = pd.read_csv(data_path)
    print(f"       Total samples: {len(df)}")

    X = df['text']
    y = df['label']

    # ========== 2. VERIFY LABELS ==========
    print(f"\n[2/7] Verifying label distribution...")
    print(f"       Label 0 (FAKE): {(y == 0).sum()}")
    print(f"       Label 1 (REAL): {(y == 1).sum()}")

    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        print("❌ ERROR: Only one class found! Check clean.py label mapping.")
        return

    # ========== 3. SPLIT DATA ==========
    print(f"\n[3/7] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Training: {len(X_train)}  |  Testing: {len(X_test)}")

    # ========== 4. TF-IDF ==========
    print(f"\n[4/7] Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"       TF-IDF shape: {X_train_tfidf.shape}")

    # ========== 5. TRAIN ==========
    print(f"\n[5/7] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_tfidf, y_train)

    # ═══ VERIFY: model.classes_ should be [0, 1] ═══
    print(f"\n       ✅ model.classes_ = {model.classes_}")
    print(f"          Index 0 → class {model.classes_[0]} ({'FAKE' if model.classes_[0] == 0 else 'REAL'})")
    print(f"          Index 1 → class {model.classes_[1]} ({'FAKE' if model.classes_[1] == 0 else 'REAL'})")

    # ========== 6. EVALUATE ==========
    print(f"\n[6/7] Evaluating model...\n")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✅ ACCURACY: {accuracy * 100:.2f}%\n")

    print("  Classification Report:")
    print("  " + "-" * 50)
    report = classification_report(
        y_test, y_pred,
        target_names=['FAKE (0)', 'REAL (1)']
    )
    for line in report.split('\n'):
        print(f"  {line}")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                    Predicted FAKE   Predicted REAL")
    print(f"  Actual FAKE         {cm[0][0]:>6}           {cm[0][1]:>6}")
    print(f"  Actual REAL         {cm[1][0]:>6}           {cm[1][1]:>6}")

    # ═══ SANITY CHECK with a known fake review ═══
    print(f"\n  ── Sanity Check ──")
    fake_test = "This is the best product ever amazing wonderful everyone should buy it now"
    real_test = "Decent product for the price. The stitching could be better and it arrived a day late."
    
    fake_vec = vectorizer.transform([fake_test])
    real_vec = vectorizer.transform([real_test])
    
    fake_pred = model.predict(fake_vec)[0]
    fake_prob = model.predict_proba(fake_vec)[0]
    real_pred = model.predict(real_vec)[0]
    real_prob = model.predict_proba(real_vec)[0]
    
    print(f'  Fake-ish review → Predicted: {"FAKE" if fake_pred == 0 else "REAL"} '
          f'(P_fake={fake_prob[0]:.2%}, P_real={fake_prob[1]:.2%})')
    print(f'  Real-ish review → Predicted: {"FAKE" if real_pred == 0 else "REAL"} '
          f'(P_fake={real_prob[0]:.2%}, P_real={real_prob[1]:.2%})')

    # ========== 7. SAVE ==========
    print(f"\n[7/7] Saving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"       ✅ model.pkl ({os.path.getsize('model.pkl')/1024:.1f} KB)")

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"       ✅ vectorizer.pkl ({os.path.getsize('vectorizer.pkl')/1024:.1f} KB)")

    # Save label mapping for reference
    label_info = {
        'classes': model.classes_.tolist(),
        'mapping': {0: 'FAKE', 1: 'REAL'},
        'predict_proba_order': ['P(FAKE)', 'P(REAL)']
    }
    with open('label_info.pkl', 'wb') as f:
        pickle.dump(label_info, f)
    print(f"       ✅ label_info.pkl saved")

    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE — Accuracy: {accuracy*100:.2f}%")
    print(f"  Labels: 0=FAKE, 1=REAL")
    print(f"  predict_proba → [P(FAKE), P(REAL)]")
    print(f"{'='*55}")

    return model, vectorizer

if __name__ == '__main__':
    train_model()
