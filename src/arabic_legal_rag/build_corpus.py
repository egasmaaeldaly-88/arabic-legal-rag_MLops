import json
import re

# ---------------------------------------------------------------------------
# PDF extraction helpers
#
# The PDF stores Arabic in *visual* (left-to-right) order, so every token has to
# be reversed. Two things break a plain reversal:
#   * lam-alef ligatures come back as one 2-character glyph already in logical
#     order ("لا", "لأ", "لإ", "لآ"); reversing the token flips them to "ال".
#   * numbers are stored left-to-right, so digit runs must not be reversed.
# ---------------------------------------------------------------------------
LIGATURES = {"\u0644\u0627", "\u0644\u0623", "\u0644\u0625", "\u0644\u0622"}  # لا لأ لإ لآ
_DIGIT_RUN = re.compile(r"([\u0660-\u0669\d]+)")
# "مادة" appears as "ةدام" in the raw text; the lookahead rejects the word
# "المادة" in the body (raw "ةداملا"), which is a cross-reference, not a heading.
AR_HEADING = re.compile(
    r"([\u0660-\u0669\d]+)\s*[\:\-]?\s*\(?\s*ةدام(?![\u0621-\u064A])\s*\)?"
)
EXPECTED_LAST_ARTICLE = 1149  # Egyptian Civil Code


def fix_ligature_chars(chars) -> int:
    """Pre-reverse ligature glyphs so the later token reversal restores them."""
    fixed = 0
    for ch in chars:
        if ch.get("text") in LIGATURES:
            ch["text"] = ch["text"][::-1]
            fixed += 1
    return fixed


def _reverse_token(token: str) -> str:
    """Reverse the letters of a visual-order token but keep digit runs readable."""
    parts = _DIGIT_RUN.split(token)  # odd indexes are digit runs
    return "".join(
        part if i % 2 == 1 else part[::-1]
        for i, part in reversed(list(enumerate(parts)))
    )


def clean_arabic_line(line_text):
    """تنظيف وعكس سطر النص العربي المستخرج من الـ PDF ليعود لاتجاهه الصحيح."""
    if not line_text:
        return ""
    line_no_english = re.sub(r'[a-zA-Z]', '', line_text)
    tokens = line_no_english.split()
    if not tokens:
        return ""
    return " ".join(_reverse_token(token) for token in reversed(tokens))


def split_page(right_text: str, page_idx: int):
    """Split a page's Arabic column into ``(leading_text, blocks)``.

    ``leading_text`` is whatever precedes the first article heading: the tail of
    the article that started on the previous page. ``blocks`` is a list of
    ``(raw_number, raw_body)``.
    """
    matches = list(AR_HEADING.finditer(right_text))
    if page_idx == 0:
        # Page 1 starts with the promulgation law (its own articles 1-2) and then
        # the Code's article 1. Keep only the Code.
        trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        ones = [i for i, m in enumerate(matches) if m.group(1).translate(trans).lstrip("0") == "1"]
        if len(ones) > 1:
            matches = matches[ones[-1]:]
    leading = right_text[: matches[0].start()] if matches else right_text
    blocks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(right_text)
        blocks.append((m.group(1), right_text[m.end():end].strip()))
    return leading, blocks


def split_article_blocks(right_text: str, page_idx: int):
    """Backward-compatible wrapper: only the article blocks of a page."""
    return split_page(right_text, page_idx)[1]


_REPEALED_EN = re.compile(
    r"Articles?\s+(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)[^.]{0,80}?repealed", re.IGNORECASE
)
# raw (visual-order) text of "المواد من ٣٨٩ إلى ٤١٧ ملغاة" is "ةاغلم ٤١٧ ىلإ ٣٨٩ نم داوملا"
_REPEALED_AR = re.compile(r"ةاغلم\s+([\u0660-\u0669\d]+)\s+ىلإ\s+([\u0660-\u0669\d]+)\s+نم")


def _to_int(digits: str) -> int:
    return int(digits.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')))


def find_repealed_ranges(left_text: str, right_text: str) -> set:
    """Ranges of repealed articles announced on a page, e.g. {(54, 80)}."""
    ranges = set()
    for m in _REPEALED_EN.finditer(left_text.replace("\n", " ")):
        a, b = int(m.group(1)), int(m.group(2))
        ranges.add((min(a, b), max(a, b)))
    for m in _REPEALED_AR.finditer(right_text):
        a, b = _to_int(m.group(1)), _to_int(m.group(2))
        ranges.add((min(a, b), max(a, b)))
    return ranges


def format_missing(present, last=EXPECTED_LAST_ARTICLE) -> str:
    """Compact list of missing article numbers, e.g. '54-80, 400'."""
    have = set(present)
    missing = [n for n in range(1, last + 1) if n not in have]
    ranges, i = [], 0
    while i < len(missing):
        j = i
        while j + 1 < len(missing) and missing[j + 1] == missing[j] + 1:
            j += 1
        ranges.append(str(missing[i]) if i == j else f"{missing[i]}-{missing[j]}")
        i = j + 1
    return f"{len(missing)} missing: " + (", ".join(ranges) if ranges else "none")


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

    import pdfplumber

    raw_articles = []  # document order: {"num", "page", "body"}
    repealed_ranges = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            current_page_num = page_idx + 1
            width = page.width
            height = page.height

            left_box = (0, 0, width / 2, height)
            right_box = (width / 2, 0, width, height)

            left_extracted = page.crop(left_box).extract_text() or ""
            right_page = page.crop(right_box)
            fix_ligature_chars(right_page.chars)  # must happen before extract_text()
            right_extracted = right_page.extract_text() or ""

            english_pages_blob += left_extracted + "\n"
            repealed_ranges |= find_repealed_ranges(left_extracted, right_extracted)

            leading, blocks = split_page(right_extracted, page_idx)
            # An article that starts at the bottom of a page continues at the top of the next
            if raw_articles and leading.strip():
                raw_articles[-1]["body"] += "\n" + leading

            for raw_num_str, raw_body in blocks:
                digits = re.findall(r'\d+', raw_num_str.strip().translate(
                    str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')))
                if not digits:
                    continue
                art_num = int(digits[0])
                if art_num == 0 or art_num > 2000:
                    continue
                raw_articles.append({"num": art_num, "page": current_page_num, "body": raw_body})

    for item in raw_articles:
        cleaned_lines = [clean_arabic_line(line) for line in item["body"].splitlines()]
        full_clean_body = clean_arabic_text(" ".join(l for l in cleaned_lines if l.strip()))
        if full_clean_body and item["num"] not in arabic_articles:
            arabic_articles[item["num"]] = {
                "text_ar": full_clean_body,
                "source_page": item["page"],
            }

    # Repealed ranges (e.g. 54-80) have no text; record them as flagged placeholders
    for first, last in sorted(repealed_ranges):
        for n in range(first, last + 1):
            if n not in arabic_articles:
                arabic_articles[n] = {
                    "text_ar": "مادة ملغاة",
                    "source_page": 0,
                    "stub": f"Articles {first}-{last} have been repealed.",
                }

    real = sum(1 for v in arabic_articles.values() if not v.get("stub"))
    print(f"Extracted {real} Arabic articles + "
          f"{len(arabic_articles) - real} repealed placeholders; "
          f"{format_missing(arabic_articles.keys())}")
    print(f"Repealed ranges found: {sorted(repealed_ranges)}")

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

        if item.get("stub"):
            metadata = classify_legal_metadata(art_num)
            final_records.append({
                "article_number": art_num,
                "book": metadata["book"], "chapter": metadata["chapter"],
                "section": metadata["section"], "topic": metadata["topic"],
                "text_ar": text_ar_val, "text_en": item["stub"],
                "is_repealed": True, "is_stub": True, "source_page": page_num,
                "citation": f"Egyptian Civil Code, Article {art_num}",
            })
            continue

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