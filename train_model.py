import json
from pathlib import Path
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
TRAIN_FILE = Path("sample_dataset/train/train.jsonl")
MODEL_OUT = Path("model/heading_classifier.joblib")

# Load training data
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(records)

# Load data
df = load_data(TRAIN_FILE)
print("Loaded:", df.shape)

if df.empty:
    print("❌ ERROR: Training file is empty.")
    exit(1)

TEXT_COL = "text"
NUMERIC_FEATURES = ["font_size", "is_bold", "is_all_caps", "x_centered"]
TARGET = "label"

# Check label distribution
label_counts = df[TARGET].value_counts()
print("Label distribution:\n", label_counts.to_string(), "\n")

# Use stratify only if all classes have ≥2 samples
use_stratify = all(label_counts >= 2)

try:
    X_train, X_val, y_train, y_val = train_test_split(
        df[[TEXT_COL] + NUMERIC_FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET] if use_stratify else None
    )
except ValueError as e:
    print(f"❌ Split error: {e}")
    print("🛠️ Tip: Add at least 2 samples per class to use stratified split.")
    exit(1)

# Preprocessing
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), "text"),
    ("num", StandardScaler(), NUMERIC_FEATURES),
])

# Classifier
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="saga",
    multi_class="multinomial"
)

# Pipeline
pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", clf)
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_val)
print("\n📊 Classification Report:\n")
print(classification_report(y_val, y_pred))

# Save model
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_OUT)
print(f"✅ Model saved to {MODEL_OUT}")
