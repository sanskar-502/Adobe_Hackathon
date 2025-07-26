#!/usr/bin/env python3
"""
Train ML model from expected outputs and schema
This script creates training data from expected JSON outputs and retrains the model
"""

import json
import re
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector

def extract_features_from_text(text, extra=None):
    """Extract features from heading text and extra span info if provided"""
    features = {
        'text': text,
        'font_size': 14.0,  # Default font size
        'is_bold': True,    # Default to bold for headings
        'is_all_caps': text.isupper(),
        'x_centered': 0.1,  # Default centered position
        'length': len(text),
        'has_numbers': bool(re.search(r'\d', text)),
        'has_dots': '.' in text,
        'starts_with_number': bool(re.match(r'^\d', text)),
        'is_known_title': any(title in text.lower() for title in [
            'revision history', 'table of contents', 'acknowledgements', 
            'references', 'introduction', 'overview'
        ]),
        'page': 0,
        'indentation': 0.0,
        'font_family': '',
        'line_spacing': 0.0
    }
    if extra:
        features.update({
            'font_size': extra.get('font_size', features['font_size']),
            'is_bold': extra.get('is_bold', features['is_bold']),
            'is_all_caps': extra.get('is_all_caps', features['is_all_caps']),
            'x_centered': extra.get('x_centered', features['x_centered']),
            'page': extra.get('page', features['page']),
            'indentation': extra.get('indentation', features['indentation']),
            'font_family': extra.get('font_family', features['font_family']),
            'line_spacing': extra.get('line_spacing', features['line_spacing'])
        })
    return features

def create_training_data_from_expected_outputs():
    """Create training data from expected outputs"""
    training_data = []
    
    # Load expected outputs
    expected_outputs_dir = Path("sample_dataset/outputs")
    
    for json_file in expected_outputs_dir.glob("*.json"):
        print(f"Processing training data from: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try to load span features if available (future-proofing)
        for heading in data.get('outline', []):
            text = heading['text'].strip()
            level = heading['level']
            page = heading['page']
            # If extra span info is present, use it
            extra = heading.get('span_features', None)
            features = extract_features_from_text(text, extra=extra)
            
            # Add page information
            features['page'] = page
            
            # Ensure all features are present
            features.setdefault('indentation', 0.0)
            features.setdefault('font_family', '')
            features.setdefault('line_spacing', 0.0)
            
            # Create training sample
            training_sample = {
                'text': features['text'],
                'font_size': features['font_size'],
                'is_bold': features['is_bold'],
                'is_all_caps': features['is_all_caps'],
                'x_centered': features['x_centered'],
                'length': features['length'],
                'has_numbers': features['has_numbers'],
                'has_dots': features['has_dots'],
                'starts_with_number': features['starts_with_number'],
                'is_known_title': features['is_known_title'],
                'page': features['page'],
                'indentation': features['indentation'],
                'font_family': features['font_family'],
                'line_spacing': features['line_spacing'],
                'label': level  # H1, H2, or NONE
            }
            
            training_data.append(training_sample)
    
    return training_data

def create_negative_samples():
    """Create negative samples (NONE class) for better training"""
    negative_samples = [
        # Common non-heading text patterns
        {'text': 'Lorem ipsum dolor sit amet', 'font_size': 10.0, 'is_bold': False, 'is_all_caps': False, 'x_centered': 0.5, 'length': 26, 'has_numbers': False, 'has_dots': False, 'starts_with_number': False, 'is_known_title': False, 'page': 1, 'indentation': 10.0, 'font_family': 'Arial', 'line_spacing': 2.0, 'label': 'NONE'},
        {'text': 'This is a paragraph of text', 'font_size': 11.0, 'is_bold': False, 'is_all_caps': False, 'x_centered': 0.4, 'length': 25, 'has_numbers': False, 'has_dots': False, 'starts_with_number': False, 'is_known_title': False, 'page': 1, 'indentation': 12.0, 'font_family': 'Arial', 'line_spacing': 2.0, 'label': 'NONE'},
        {'text': 'Page 1 of 10', 'font_size': 9.0, 'is_bold': False, 'is_all_caps': False, 'x_centered': 0.8, 'length': 12, 'has_numbers': True, 'has_dots': False, 'starts_with_number': False, 'is_known_title': False, 'page': 1, 'indentation': 8.0, 'font_family': 'Arial', 'line_spacing': 2.0, 'label': 'NONE'},
        {'text': 'Footer text', 'font_size': 8.0, 'is_bold': False, 'is_all_caps': False, 'x_centered': 0.6, 'length': 11, 'has_numbers': False, 'has_dots': False, 'starts_with_number': False, 'is_known_title': False, 'page': 1, 'indentation': 8.0, 'font_family': 'Arial', 'line_spacing': 2.0, 'label': 'NONE'},
    ]
    # Ensure all features are present
    for sample in negative_samples:
        sample.setdefault('indentation', 0.0)
        sample.setdefault('font_family', '')
        sample.setdefault('line_spacing', 0.0)
    return negative_samples

def train_model():
    """Train the ML model on expected outputs"""
    print("Creating training data from expected outputs...")
    
    # Get training data from expected outputs
    training_data = create_training_data_from_expected_outputs()
    
    # Add negative samples
    negative_samples = create_negative_samples()
    training_data.extend(negative_samples)
    
    if not training_data:
        print("No training data found!")
        return False
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    # Ensure all required columns exist
    required_cols = [
        'text', 'font_size', 'is_bold', 'is_all_caps', 'x_centered', 'length', 'has_numbers', 'has_dots',
        'starts_with_number', 'is_known_title', 'page', 'indentation', 'font_family', 'line_spacing', 'label'
    ]
    for col in required_cols:
        if col not in df.columns:
            if col == 'font_family':
                df[col] = ''
            elif col == 'label':
                df[col] = 'NONE'
            else:
                df[col] = 0.0
    # Ensure font_family is always a string
    df['font_family'] = df['font_family'].astype(str)
    print(f"DataFrame columns: {list(df.columns)}")
    print(df.head())
    print(df.dtypes)
    # Reorder columns to match expected order
    ordered_cols = [
        'text', 'font_size', 'is_bold', 'is_all_caps', 'x_centered', 'length', 'has_numbers', 'has_dots',
        'starts_with_number', 'is_known_title', 'page', 'indentation', 'font_family', 'line_spacing', 'label'
    ]
    df = df[ordered_cols]
    print(f"Training data shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Feature columns
    TEXT_COL = "text"
    NUMERIC_FEATURES = ["font_size", "is_bold", "is_all_caps", "x_centered", "length", "has_numbers", "has_dots", "starts_with_number", "is_known_title", "page", "indentation", "line_spacing"]
    CATEGORICAL_FEATURES = ["font_family"]
    TARGET = "label"
    
    # Check if we have enough samples for each class
    label_counts = df[TARGET].value_counts()
    print(f"\nLabel distribution:\n{label_counts}")
    
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
        print(f"Split error: {e}")
        print("Adding more samples per class...")
        return False
    
    # Preprocessing
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), "text"),
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("font_family", OneHotEncoder(handle_unknown='ignore'), make_column_selector(pattern='font_family')),
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
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save model
    model_path = Path("model/heading_classifier.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
    return True

def main():
    """Main training function"""
    print("Training ML model from expected outputs...")
    print("=" * 50)
    
    # Check if expected outputs exist
    expected_outputs_dir = Path("sample_dataset/outputs")
    if not expected_outputs_dir.exists():
        print("No expected outputs found in sample_dataset/outputs/")
        print("Please run the processing first to generate expected outputs.")
        return
    
    # Train model
    success = train_model()
    
    if success:
        print("\nTraining completed successfully!")
        print("The model has been updated with your expected outputs.")
        print("You can now add more training data and retrain as needed.")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 