"""
clean.py - Data Preprocessing Script
Cleans the raw reviews.csv dataset for ML training.

LABELS:
  0 = FAKE  (Computer Generated - CG)
  1 = REAL  (Original - OR)
"""

import pandas as pd
import re

def clean_text(text):
    """Remove special characters, extra spaces, and lowercase the text."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(input_path='reviews.csv', output_path='reviews_clean.csv'):
    """Load, clean, and save the dataset."""

    print("[1/6] Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"       Raw dataset shape: {df.shape}")
    print(f"       Columns found: {list(df.columns)}")

    # ── Keep only the columns we need ──
    if 'text_' in df.columns:
        df = df[['text_', 'label']].copy()
        df.rename(columns={'text_': 'text'}, inplace=True)
    elif 'text' in df.columns:
        df = df[['text', 'label']].copy()
    else:
        df = df.iloc[:, [0, -1]].copy()
        df.columns = ['text', 'label']

    # ── Show raw label distribution ──
    print(f"\n[2/6] Raw label distribution:")
    print(df['label'].value_counts())

    # ── Drop null / empty values ──
    print("\n[3/6] Removing null and empty values...")
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['text'].str.strip() != '']

    # ══════════════════════════════════════════════
    #  CRITICAL FIX: Label Mapping
    #  CG (Computer Generated) = FAKE  = 0
    #  OR (Original)           = REAL  = 1
    #
    #  We use:  FAKE=0, REAL=1
    #  So predict_proba returns [P(fake), P(real)]
    #  And prediction: 0=FAKE, 1=REAL
    # ══════════════════════════════════════════════
    print("[4/6] Mapping labels (CG=0/Fake, OR=1/Real)...")

    label_map = {
        'CG': 0,     # Computer Generated = FAKE = 0
        'OR': 1,     # Original           = REAL = 1
    }
    df['label'] = df['label'].map(label_map)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # ── Clean review text ──
    print("[5/6] Cleaning text...")
    df['text'] = df['text'].apply(clean_text)

    # Remove very short reviews
    df = df[df['text'].apply(lambda x: len(x.split()) >= 5)]

    # Remove duplicates
    df.drop_duplicates(subset=['text'], inplace=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Save ──
    print(f"[6/6] Saving cleaned dataset to '{output_path}'...")
    df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"  CLEANING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total samples : {len(df)}")
    print(f"  FAKE (0 / CG) : {(df['label'] == 0).sum()}")
    print(f"  REAL (1 / OR) : {(df['label'] == 1).sum()}")
    print(f"{'='*50}")

    return df

if __name__ == '__main__':
    preprocess_dataset()
