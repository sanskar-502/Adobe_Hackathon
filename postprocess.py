def clean_outline(outline):
    """
    Remove duplicate text/page combinations, filter out junk headings.
    """
    seen = set()
    cleaned = []
    for o in outline:
        key = (o["text"].strip(), o["page"])
        if key in seen:
            continue
        seen.add(key)

        if len(o["text"]) < 4 or len(o["text"]) > 120:
            continue

        cleaned.append(o)
    return cleaned


def promote_headings(spans, predicted_outline, h1_threshold=0.95, h2_threshold=0.85):
    """
    Promote spans to headings if they're highly centered + bold + upper font size,
    even if the model missed them.
    """
    for span in spans:
        text = span["text"]
        if any(text == h["text"] and span["page"] == h["page"] for h in predicted_outline):
            continue

        if span["x_centered"] < 0.1 and span["is_bold"] and span["font_size"] > 14:
            predicted_outline.append({
                "level": "H2",
                "text": text,
                "page": span["page"]
            })
    return predicted_outline
