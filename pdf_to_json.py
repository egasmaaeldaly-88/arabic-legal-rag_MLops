import json
import re
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display

def convert_arabic_pdf_to_json(pdf_path, output_json_path):
    raw_text = ""
    
    # 1. Extract raw text page by page
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + "\n"

    # 2. Fix the backward Arabic text
    reshaped_text = arabic_reshaper.reshape(raw_text)
    corrected_text = get_display(reshaped_text)

    # 3. Split text by article patterns (now searching the corrected text)
    article_pattern = r"(مادة\s*\(?\d+\)?|المادة\s*\d+)"
    parts = re.split(article_pattern, corrected_text)

    legal_articles = []

    # 4. Process the split text into structured dictionaries
    for i in range(1, len(parts), 2):
        article_header = parts[i].strip()
        article_text = parts[i+1].strip() if i + 1 < len(parts) else ""
        
        # Clean up extra newlines and spaces within the text
        clean_text = " ".join(article_text.split())

        legal_articles.append({
            "article_number": article_header,
            "text": clean_text,
            "source_document": pdf_path
        })

    # 5. Export to JSON with UTF-8 encoding for Arabic characters
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(legal_articles, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete! Saved {len(legal_articles)} articles to {output_json_path}")

# Execute the function
if __name__ == "__main__":
    convert_arabic_pdf_to_json("egyptian_civil_code.pdf", "data/legal_corpus.json")