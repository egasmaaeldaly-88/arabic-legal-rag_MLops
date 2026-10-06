import json
import re
import pdfplumber

def clean_arabic_line(line_text):
    """
    Cleans a line of text extracted from reversed PDF streams:
    1. Removes reversed English text columns.
    2. Reverses character order so Arabic words read correctly.
    """
    if not line_text:
        return ""
    # Remove English characters/words
    line_no_english = re.sub(r'[a-zA-Z]', '', line_text)
    
    tokens = line_no_english.split()
    if not tokens:
        return ""
        
    cleaned_line = " ".join([token[::-1] for token in reversed(tokens)])
    return cleaned_line

def clean_arabic_text(text: str) -> str:
    """
    Cleans residual extraction noise (punctuation, brackets, stray spaces).
    """
    if not text:
        return ""

    text = re.sub(r'\s*\)\s*\(\s*', ' ', text)
    text = re.sub(r'[\(\)]', ' ', text)
    text = re.sub(r'[\,\'\;\؛]+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([\.،؛:])', r'\1', text)
    return text.strip(' .,،;')

def arabic_to_int(arabic_num_str: str) -> int:
    """Converts Arabic-Indic numerals or mixed strings to a clean integer."""
    trans_table = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    cleaned_str = arabic_num_str.translate(trans_table)
    digits = re.findall(r'\d+', cleaned_str)
    if digits:
        return int(digits[0])
    return 0

def clean_duplicate_articles(data):
    """
    Removes duplicate articles based on article_number, 
    keeping the record with the most complete text/metadata.
    """
    seen_articles = {}
    
    for item in data:
        article_num = item.get("article_number")
        text = item.get("ar_text") or item.get("text_ar", "")
        
        if article_num is None:
            continue
            
        # If the article is already seen, keep the one with longer/richer text
        if article_num in seen_articles:
            existing_item = seen_articles[article_num]
            existing_text = existing_item.get("ar_text") or existing_item.get("text_ar", "")
            if len(text) > len(existing_text):
                seen_articles[article_num] = item
        else:
            seen_articles[article_num] = item
            
    # Convert back to list and sort by article number if possible
    cleaned_data = list(seen_articles.values())
    try:
        cleaned_data.sort(key=lambda x: int(x.get("article_number", 0)))
    except ValueError:
        pass
        
    print(f"Deduplication complete: Reduced from {len(data)} to {len(cleaned_data)} unique articles.")
    return cleaned_data

def build_complete_legal_corpus(pdf_path, output_json_path):
    raw_text = ""
    
    print("Step 1: Reading PDF file...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + f"\n[PAGE_{page_idx+1}]\n"

    print("Step 2: Parsing and cleaning articles...")
    reversed_article_pattern = r"([\u0660-\u0669\d]+\s*[\:\-]?\s*\(?\s*ةدام\s*\)?)"
    parts = re.split(reversed_article_pattern, raw_text)

    legal_articles = []

    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            raw_header = parts[i].strip()
            raw_body = parts[i+1].strip() if i + 1 < len(parts) else ""

            # Fix header and extract integer article number
            clean_header = clean_arabic_line(raw_header)
            art_num = arabic_to_int(clean_header)
            
            # Skip if article number couldn't be parsed properly
            if art_num == 0:
                continue

            # Fix body line by line
            lines = raw_body.splitlines()
            cleaned_body_lines = [clean_arabic_line(line) for line in lines]
            full_clean_body = " ".join([line for line in cleaned_body_lines if line.strip()])
            full_clean_body = clean_arabic_text(full_clean_body)

            # Check if article is repealed
            is_rep = "ملغاة" in full_clean_body or "ملغى" in full_clean_body

            # Target Schema Structure matching the Handbook
            article_record = {
                "article_number": art_num,
                "book": "الأحكام العامة / القانون المدني",
                "chapter": "",
                "section": "",
                "topic": "",
                "ar_text": full_clean_body,
                "text_en": "",
                "is_repealed": is_rep,
                "source_page": 1,
                "citation": f"Egyptian Civil Code, Article {art_num}"
            }

            legal_articles.append(article_record)

    # Step 3: Clean duplicate articles
    print("Step 3: Removing duplicate articles...")
    cleaned_articles = clean_duplicate_articles(legal_articles)

    # Step 4: Export structured JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)

    print(f"Step 4: Success! Processed and saved {len(cleaned_articles)} unique structured articles to '{output_json_path}'.")

if __name__ == "__main__":
    PDF_INPUT = "data/egyptian_civil_code.pdf"
    JSON_OUTPUT = "data/legal_corpus_structured.json"
    
    build_complete_legal_corpus(PDF_INPUT, JSON_OUTPUT)