#!/usr/bin/env python3
"""
Add training data for new PDFs
This script helps you add new PDF files and their expected outputs for training
"""

import json
import shutil
from pathlib import Path

def add_new_training_data(pdf_file_path, expected_output):
    """
    Add a new PDF and its expected output for training
    
    Args:
        pdf_file_path: Path to the PDF file
        expected_output: Dictionary with expected JSON output
    """
    
    # Copy PDF to sample_dataset/pdfs/
    pdfs_dir = Path("sample_dataset/pdfs")
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = Path(pdf_file_path).name
    target_pdf_path = pdfs_dir / pdf_name
    
    try:
        shutil.copy2(pdf_file_path, target_pdf_path)
        print(f"✅ Copied {pdf_name} to sample_dataset/pdfs/")
    except Exception as e:
        print(f"❌ Error copying PDF: {e}")
        return False
    
    # Save expected output to sample_dataset/outputs/
    outputs_dir = Path("sample_dataset/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_stem = Path(pdf_file_path).stem
    output_file = outputs_dir / f"{pdf_stem}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(expected_output, f, indent=2)
        print(f"✅ Saved expected output to {output_file}")
    except Exception as e:
        print(f"❌ Error saving expected output: {e}")
        return False
    
    return True

def create_template():
    """Create a template for expected output"""
    template = {
        "title": "Your Document Title",
        "outline": [
            {
                "level": "H1",
                "text": "1. First Main Heading",
                "page": 1
            },
            {
                "level": "H2",
                "text": "1.1 First Sub Heading",
                "page": 1
            },
            {
                "level": "H2",
                "text": "1.2 Second Sub Heading",
                "page": 2
            },
            {
                "level": "H1",
                "text": "2. Second Main Heading",
                "page": 3
            }
        ]
    }
    return template

def main():
    """Main function to add training data"""
    print("Add Training Data for New PDFs")
    print("=" * 40)
    
    print("\nTo add new training data:")
    print("1. Place your PDF file in the current directory")
    print("2. Create the expected output JSON")
    print("3. Run this script with the file paths")
    
    print("\nExample usage:")
    print("python add_training_data.py")
    
    print("\nTemplate for expected output:")
    template = create_template()
    print(json.dumps(template, indent=2))
    
    print("\nSteps to add new training data:")
    print("1. Copy your PDF to sample_dataset/pdfs/")
    print("2. Create expected output JSON in sample_dataset/outputs/")
    print("3. Run: python train_from_expected_outputs.py")
    print("4. Test with: python process_pdfs.py")

if __name__ == "__main__":
    main() 