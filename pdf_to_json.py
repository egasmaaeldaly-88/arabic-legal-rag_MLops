import json
import re
import pdfplumber

def clean_arabic_line(line_text):
    """
    Cleans a line of text extracted from reversed PDF streams:
    1. Removes reversed English text columns (e.g. 'sthgiR dna swaL').
    2. Reverses character order so Arabic words read left-to-right correctly.
    """
    # Remove English characters/words (which are upside down translations in this PDF)
    line_no_english = re.sub(r'[a-zA-Z]', '', line_text)
    
    # Split by whitespace, clean empty tokens
    tokens = line_no_english.split()
    if not tokens:
        return ""
        
    # Re-assemble line with proper word and character reversal
    # Flipped at token level to maintain word-level integrity
    cleaned_line = " ".join([token[::-1] for token in reversed(tokens)])
    return cleaned_line

def convert_arabic_pdf_to_json(pdf_path, output_json_path):
    raw_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + "\n"

    # Match reversed article headers like '١ ةدام' or 'ةدام'
    reversed_article_pattern = r"([\u0660-\u0669\d]+\s*[\:\-]?\s*\(?\s*ةدام\s*\)?)"
    parts = re.split(reversed_article_pattern, raw_text)

    legal_articles = []

    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            raw_header = parts[i].strip()
            raw_body = parts[i+1].strip() if i + 1 < len(parts) else ""

            # Fix header (e.g., '١ ةدام' -> 'مادة ١')
            clean_header = clean_arabic_line(raw_header)
            
            # Fix body line by line
            lines = raw_body.splitlines()
            cleaned_body_lines = [clean_arabic_line(line) for line in lines]
            
            # Remove empty lines and join into cohesive text
            full_clean_body = " ".join([line for line in cleaned_body_lines if line.strip()])

            legal_articles.append({
                "article_number": clean_header,
                "text": full_clean_body,
                "source_document": pdf_path
            })

    # Export clean output
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(legal_articles, f, ensure_ascii=False, indent=2)

    print(f"Refined extraction complete! Saved {len(legal_articles)} clean articles to {output_json_path}")

if __name__ == "__main__":
    convert_arabic_pdf_to_json("data/egyptian_civil_code.pdf", "data/legal_corpus.json")