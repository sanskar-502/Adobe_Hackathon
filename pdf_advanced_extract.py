import pdfplumber
import pytesseract
from PIL import Image
import io
from pathlib import Path

def extract_spans(pdf_path):
    spans = []
    font_sizes = []
    pdf_path = str(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # --- Extract text spans ---
            for char in page.chars:
                text = char.get('text', '').strip()
                if not text:
                    continue
                font_size = float(char.get('size', 0))
                font_name = char.get('fontname', '')
                x0, top, x1, bottom = char['x0'], char['top'], char['x1'], char['bottom']
                is_bold = 'Bold' in font_name
                is_all_caps = text.isupper()
                x_centered = abs(((x0 + x1) / 2) - (page.width / 2)) / page.width
                indentation = x0
                line_spacing = 0.0  # Not available at char level
                spans.append({
                    'text': text,
                    'font_size': font_size,
                    'is_bold': is_bold,
                    'is_all_caps': is_all_caps,
                    'x_centered': x_centered,
                    'indentation': indentation,
                    'font_family': font_name,
                    'line_spacing': line_spacing,
                    'page': page_num,
                    'source': 'text'
                })
                font_sizes.append(font_size)
            # --- Extract images and run OCR ---
            for img_dict in page.images:
                try:
                    img_bytes = page.extract_image(img_dict['object_id'])['image']
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img)
                    for line in ocr_text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        spans.append({
                            'text': line,
                            'font_size': 12.0,  # Default for OCR
                            'is_bold': False,
                            'is_all_caps': line.isupper(),
                            'x_centered': 0.5,
                            'indentation': 0.0,
                            'font_family': 'OCR',
                            'line_spacing': 0.0,
                            'page': page_num,
                            'source': 'ocr_image'
                        })
                        font_sizes.append(12.0)
                except Exception as e:
                    print(f"OCR image extraction failed on page {page_num}: {e}")
    return spans, font_sizes

if __name__ == "__main__":
    # Quick test
    pdf_path = Path("sample_dataset/pdfs/file03.pdf")
    spans, font_sizes = extract_spans(pdf_path)
    print(f"Extracted {len(spans)} spans, font sizes: {set(font_sizes)}")
    for s in spans[:10]:
        print(s) 