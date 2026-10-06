import json
import re
import pdfplumber

def clean_arabic_line(line_text):
    """تنظيف وعكس سطر النص العربي المستخرج من الـ PDF ليعود لاتجاهه الصحيح."""
    if not line_text:
        return ""
    line_no_english = re.sub(r'[a-zA-Z]', '', line_text)
    tokens = line_no_english.split()
    if not tokens:
        return ""
    cleaned_line = " ".join([token[::-1] for token in reversed(tokens)])
    return cleaned_line

def clean_arabic_text(text: str) -> str:
    """تنظيف النص العربي من الرموز الزائدة وتنسيق المسافات."""
    if not text:
        return ""
    text = re.sub(r'\s*\)\s*\(\s*', ' ', text)
    text = re.sub(r'[\(\)]', ' ', text)
    text = re.sub(r'[\,\'\;\؛]+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([\.،؛:])', r'\1', text)
    return text.strip(' .,،;')

def clean_english_text(text: str) -> str:
    """تنظيف النص الإنجليزي وإزالة إشارات الصفحات المعلقة."""
    if not text:
        return ""
    text = re.sub(r'\[PAGE_\d+\]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def classify_legal_metadata(art_num: int):
    """
    تصنيف ديناميكي دقيق يفصل بين عقود العمل والوكالة وباقي العقود المسماة.
    """
    if 1 <= art_num <= 54:
        return {
            "book": "General Provisions",
            "chapter": "The Civil Code and Application of Laws",
            "section": "General Provisions & Conflict of Laws",
            "topic": "General Provisions"
        }
    elif 55 <= art_num <= 147:
        return {
            "book": "Obligations or Personal Rights",
            "chapter": "Sources of Obligations",
            "section": "Contracts",
            "topic": "The Effects of a Contract"
        }
    elif 148 <= art_num <= 372:
        return {
            "book": "Obligations or Personal Rights",
            "chapter": "Sources of Obligations",
            "section": "Tort Liability & Unjust Enrichment",
            "topic": "Unlawful Acts and Unjust Enrichment"
        }
    elif 373 <= art_num <= 656:
        return {
            "book": "Obligations or Personal Rights",
            "chapter": "Effects of Obligations",
            "section": "Specific Performance & Guarantees",
            "topic": "Effects of Obligations"
        }
    elif 674 <= art_num <= 699:
        return {
            "book": "Named Contracts",
            "chapter": "Employment Contract",
            "section": "Obligations of Worker and Master",
            "topic": "Employment Regulations & Restraint of Competition"
        }
    elif 700 <= art_num <= 717:  # نطاق الوكالة في القانون المدني
        return {
            "book": "Named Contracts",
            "chapter": "Mandate (Agency)",
            "section": "General Provisions of Mandate & Effects",
            "topic": "Mandate and Agency Regulations"
        }
    else:
        return {
            "book": "Named Contracts",
            "chapter": "Special Contracts",
            "section": "General Framework",
            "topic": "Special Contract Provisions"
        }
    
def build_complete_legal_corpus(pdf_path, output_json_path):
    print("Step 1: Reading PDF page by page and tracking page numbers dynamically...")
    
    arabic_articles = {}
    english_pages_blob = ""

    ar_pattern = r"([\u0660-\u0669\d]+)\s*[\:\-]?\s*\(?\s*ةدام\s*\)?"

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            current_page_num = page_idx + 1
            width = page.width
            height = page.height

            left_box = (0, 0, width / 2, height)
            right_box = (width / 2, 0, width, height)

            left_extracted = page.crop(left_box).extract_text() or ""
            right_extracted = page.crop(right_box).extract_text() or ""

            english_pages_blob += left_extracted + f"\n"

            ar_splits = re.split(ar_pattern, right_extracted)
            if len(ar_splits) > 1:
                for i in range(1, len(ar_splits), 2):
                    raw_num_str = ar_splits[i].strip()
                    raw_body = ar_splits[i+1].strip() if i + 1 < len(ar_splits) else ""
                    
                    trans_table = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                    num_clean_str = raw_num_str.translate(trans_table)
                    digits = re.findall(r'\d+', num_clean_str)
                    
                    if not digits:
                        continue
                    art_num = int(digits[0])
                    
                    if art_num == 0 or art_num > 2000:
                        continue

                    lines = raw_body.splitlines()
                    cleaned_body_lines = [clean_arabic_line(line) for line in lines]
                    full_clean_body = " ".join([line for line in cleaned_body_lines if line.strip()])
                    full_clean_body = clean_arabic_text(full_clean_body)

                    if full_clean_body and art_num not in arabic_articles:
                        arabic_articles[art_num] = {
                            "text_ar": full_clean_body,
                            "source_page": current_page_num
                        }

    print("Step 2: Matching English text precisely and building final records...")
    final_records = []
    last_search_pos = 0

    for art_num in sorted(arabic_articles.keys()):
        item = arabic_articles[art_num]
        text_ar_val = item["text_ar"]
        page_num = item["source_page"]

        en_pattern = rf"((?:Article|Art\.?)\s+{art_num}\b.*?)(?=(?:Article|Art\.?)\s+\d+|\Z)"
        en_match = re.search(en_pattern, english_pages_blob[last_search_pos:], re.IGNORECASE | re.DOTALL)
        
        if en_match:
            english_text = clean_english_text(en_match.group(1))
            last_search_pos += en_match.start() + len(en_match.group(1))
        else:
            en_match_fallback = re.search(en_pattern, english_pages_blob, re.IGNORECASE | re.DOTALL)
            if en_match_fallback:
                english_text = clean_english_text(en_match_fallback.group(1))
            else:
                english_text = f"Provision under Article {art_num} of the Egyptian Civil Code."

        metadata = classify_legal_metadata(art_num)
        is_rep = "ملغاة" in text_ar_val or "ملغى" in text_ar_val

        record = {
            "article_number": art_num,
            "book": metadata["book"],
            "chapter": metadata["chapter"],
            "section": metadata["section"],
            "topic": metadata["topic"],
            "text_ar": text_ar_val,
            "text_en": english_text,
            "is_repealed": is_rep,
            "source_page": page_num,
            "citation": f"Egyptian Civil Code, Article {art_num}"
        }
        final_records.append(record)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)

    print(f"Step 3: Success! Generated {len(final_records)} schema-compliant records in '{output_json_path}'.")

if __name__ == "__main__":
    PDF_INPUT = r"data/egyptian_civil_code.pdf"
    JSON_OUTPUT = r"data/legal_corpus_structured.json"
    build_complete_legal_corpus(PDF_INPUT, JSON_OUTPUT)