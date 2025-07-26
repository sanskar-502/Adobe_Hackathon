from pathlib import Path
import joblib
import json
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import re
from postprocess import clean_outline, promote_headings
from pdf_extract import extract_spans as extract_spans_pymupdf
from pdf_advanced_extract import extract_spans as extract_spans_pdfplumber

# Extraction mode: 'pymupdf', 'pdfplumber', or 'both'
EXTRACTION_MODE = 'both'

# Load model
MODEL_PATH = Path("model/heading_classifier.joblib")
model = joblib.load(MODEL_PATH)

def get_spans(pdf_path):
    if EXTRACTION_MODE == 'pymupdf':
        return extract_spans_pymupdf(pdf_path)
    elif EXTRACTION_MODE == 'pdfplumber':
        return extract_spans_pdfplumber(pdf_path)
    elif EXTRACTION_MODE == 'both':
        spans1, font_sizes1 = extract_spans_pymupdf(pdf_path)
        spans2, font_sizes2 = extract_spans_pdfplumber(pdf_path)
        # Merge and deduplicate spans by (text, page, font_size, indentation)
        seen = set()
        merged_spans = []
        for s in spans1 + spans2:
            key = (s['text'], s['page'], s.get('font_size', 0), s.get('indentation', 0.0))
            if key not in seen:
                seen.add(key)
                merged_spans.append(s)
        merged_font_sizes = font_sizes1 + font_sizes2
        return merged_spans, merged_font_sizes
    else:
        raise ValueError(f"Unknown extraction mode: {EXTRACTION_MODE}")

def predict_headings_ml(spans):
    if not spans:
        return []
    features = []
    for span in spans:
        features.append({
            'text': span['text'],
            'font_size': span['font_size'],
            'is_bold': span['is_bold'],
            'is_all_caps': span['is_all_caps'],
            'x_centered': span['x_centered'],
            'length': len(span['text']),
            'has_numbers': bool(re.search(r'\d', span['text'])),
            'has_dots': '.' in span['text'],
            'starts_with_number': bool(re.match(r'^\d', span['text'])),
            'is_known_title': any(title in span['text'].lower() for title in [
                'revision history', 'table of contents', 'acknowledgements', 
                'references', 'introduction', 'overview'
            ]),
            'page': span['page'],
            'indentation': span.get('indentation', 0.0),
            'font_family': span.get('font_family', ''),
            'line_spacing': span.get('line_spacing', 0.0)
        })
    df = pd.DataFrame(features)
    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        outline = []
        for i, (span, pred, prob) in enumerate(zip(spans, predictions, probabilities)):
            if pred in ['H1', 'H2', 'H3', 'H4'] and max(prob) > 0.6:
                outline.append({
                    'level': pred,
                    'text': span['text'],
                    'page': span['page']
                })
        return outline
    except Exception as e:
        print(f"ML prediction failed: {e}")
        return []

def process_pdf_ml_enhanced(pdf_path, output_path):
    print(f"Processing: {pdf_path.name}")
    # Use the selected extraction method(s)
    spans, font_sizes = get_spans(pdf_path)
    ml_outline = predict_headings_ml(spans)
    if len(ml_outline) < 3:
        print("ML found few headings, using rule-based fallback...")
        from pdf_extract import assign_headings
        rule_outline = assign_headings(spans, font_sizes)
        outline = rule_outline
    else:
        outline = ml_outline
    outline = clean_outline(outline)
    outline = promote_headings(spans, outline)
    if not outline:
        print("No headings found, outputting empty outline.")
        outline = []
    # Extract title from largest text on first page
    first_page_spans = [s for s in spans if s["page"] == 0]
    if first_page_spans:
        max_font = max(s["font_size"] for s in first_page_spans)
        title_spans = [s for s in first_page_spans if s["font_size"] == max_font]
        title_text = "  ".join([s["text"] for s in title_spans]).strip() + "  "
    else:
        title_text = "Document Title"
    result = {
        "title": title_text,
        "outline": outline
    }
    if len(result["outline"]) == 1 and result["outline"][0]["text"].strip() == result["title"].strip():
        print("Only heading is identical to title, removing from outline.")
        result["outline"] = []
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Processed: {pdf_path.name} -> {output_path.name}")
    print(f"   Found {len(result['outline'])} headings")

def main():
    input_dir = Path("sample_dataset/pdfs")
    output_dir = Path("sample_dataset/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in sample_dataset/pdfs/")
        return
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}.json"
        try:
            process_pdf_ml_enhanced(pdf_file, output_file)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
