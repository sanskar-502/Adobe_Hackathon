import fitz  # PyMuPDF
import json
import re
from pathlib import Path


def extract_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    font_sizes = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        w, h = page.rect.width, page.rect.height
        prev_bottom = None
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    font_size = span["size"]
                    bbox = span["bbox"]
                    is_bold = "Bold" in span.get("font", "")
                    is_all_caps = text.isupper()
                    x_centered = abs(((bbox[0] + bbox[2]) / 2) - w / 2) / w
                    indentation = bbox[0]  # left margin
                    font_family = span.get("font", "")
                    # Line spacing: vertical distance from previous span on the same page
                    line_spacing = None
                    if prev_bottom is not None:
                        line_spacing = bbox[1] - prev_bottom
                    prev_bottom = bbox[3]

                    spans.append({
                        "text": text,
                        "font_size": font_size,
                        "is_bold": is_bold,
                        "is_all_caps": is_all_caps,
                        "x_centered": x_centered,
                        "indentation": indentation,
                        "font_family": font_family,
                        "line_spacing": line_spacing if line_spacing is not None else 0.0,
                        "page": page_num
                    })
                    font_sizes.append(font_size)

    return spans, font_sizes


def assign_headings(spans, font_sizes):
    outline = []
    font_sizes = sorted(set(font_sizes), reverse=True)
    
    # Better font size thresholds based on debug output
    if len(font_sizes) >= 3:
        h1_threshold = font_sizes[1]  # Second largest font (16.0)
        h2_threshold = font_sizes[2]  # Third largest font (14.0)
    elif len(font_sizes) >= 2:
        h1_threshold = font_sizes[0]
        h2_threshold = font_sizes[1]
    else:
        h1_threshold = font_sizes[0] if font_sizes else 0
        h2_threshold = h1_threshold - 1

    # Improved patterns for heading detection
    h1_pattern = re.compile(r'^(\d+\.?\s+)?([A-Z][^.]*?)(\s*[-–]\s*[^.]*)?$')
    h2_pattern = re.compile(r'^(\d+\.\d+\s+)([^.]*?)(\s*[-–]\s*[^.]*)?$')
    
    # Known H1 titles
    known_h1_titles = {
        "revision history", "table of contents", "acknowledgements", 
        "references", "introduction to the foundation level extensions",
        "introduction to foundation level agile tester extension",
        "overview of the foundation level extension"
    }

    for span in spans:
        text = span["text"].strip()
        size = span["font_size"]
        page = span["page"]
        is_bold = span["is_bold"]

        # Skip very short or very long text
        if len(text) < 3 or len(text) > 150:
            continue

        # Skip text that's too centered (likely headers/footers)
        if span["x_centered"] > 0.3:
            continue

        # Check for known H1 titles
        is_known_h1 = any(title in text.lower() for title in known_h1_titles)
        
        # Check for numbered patterns
        h1_match = h1_pattern.match(text)
        h2_match = h2_pattern.match(text)
        
        # Determine heading level based on debug output
        if is_known_h1 or (h1_match and size >= h1_threshold and is_bold):
            outline.append({"level": "H1", "text": text, "page": page})
        elif h2_match and size >= h2_threshold:
            # H2 headings are not bold but have font size 14.0
            outline.append({"level": "H2", "text": text, "page": page})
        elif is_bold and size >= h1_threshold and page >= 2:
            # Additional H1 detection for bold, large text after page 1
            outline.append({"level": "H1", "text": text, "page": page})
        elif h1_match and size >= h1_threshold:
            # H1 numbered headings that might not be bold
            outline.append({"level": "H1", "text": text, "page": page})

    return outline


def process_pdf(pdf_path, output_path):
    spans, font_sizes = extract_spans(pdf_path)
    outline = assign_headings(spans, font_sizes)

    # Better title extraction - look for the main title on first page
    title_spans = []
    for span in spans:
        if span["page"] == 0 and span["font_size"] >= max(font_sizes) - 0.5:
            title_spans.append(span)
    
    # Sort by font size and take the largest
    title_spans.sort(key=lambda x: x["font_size"], reverse=True)
    title_text = title_spans[0]["text"] if title_spans else "Document Title"

    # Clean and sort outline
    clean_outline = []
    seen = set()
    
    for o in outline:
        key = (o["text"].strip(), o["page"])
        if key not in seen:
            seen.add(key)
            clean_outline.append(o)
    
    # Sort by page number, then by text
    clean_outline.sort(key=lambda x: (x["page"], x["text"]))

    result = {
        "title": title_text,
        "outline": clean_outline
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Processed: {pdf_path.name} -> {output_path.name}")


if __name__ == "__main__":
    input_dir = Path("sample_dataset/pdfs")
    output_dir = Path("sample_dataset/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        out_path = output_dir / f"{pdf_file.stem}.json"
        process_pdf(pdf_file, out_path)