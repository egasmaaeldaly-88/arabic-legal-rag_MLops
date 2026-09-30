import json
import re

def clean_arabic_text(text: str) -> str:
    """
    Cleans up residual extraction noise (redundant punctuation, broken brackets,
    and stray spaces) without altering correctly-oriented text or numbers.
    """
    if not text:
        return ""

    # 1. Remove floating / unmatched parentheses patterns like ')(', ') ('
    text = re.sub(r'\s*\)\s*\(\s*', ' ', text)
    
    # 2. Clean loose brackets while preserving meaningful layout
    text = re.sub(r'[\(\)]', ' ', text)

    # 3. Remove stacked or stray punctuation (e.g., ', , .', ';', floating quotes)
    text = re.sub(r'[\,\'\;\؛]+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)

    # 4. Collapse extra whitespace and fix space before punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([\.،؛:])', r'\1', text)

    # 5. Trim leading/trailing stray symbols
    return text.strip(' .,،;')


def post_process_cleaned_corpus(input_path: str, output_path: str):
    """
    Reads the output from the 2nd script, applies final punctuation and formatting fixes,
    and saves the polished result.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_path}'. Make sure the 2nd script has run successfully.")
        return

    cleaned_dataset = []

    for item in dataset:
        header = item.get("article_number", "")
        body = item.get("text", "")

        # 1. Clean header and normalize format to "مادة X"
        header_cleaned = clean_arabic_text(header)
        header_cleaned = re.sub(
            r'مادة\s*\(?\s*([\u0660-\u0669\d]+)\s*\)?', 
            r'مادة \1', 
            header_cleaned
        )

        # 2. Clean main text body
        body_cleaned = clean_arabic_text(body)

        cleaned_dataset.append({
            "article_number": header_cleaned,
            "text": body_cleaned,
            "source_document": item.get("source_document", "")
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_dataset, f, ensure_ascii=False, indent=2)

    print(f"Pipeline Step 3 Complete: Processed {len(cleaned_dataset)} articles from '{input_path}' and updated '{output_path}'.")


if __name__ == "__main__":
    # Reads the output of the 2nd script and overwrites it with the final polished version
    INPUT_FILE = "data/legal_corpus_cleaned.json"
    OUTPUT_FILE = "data/legal_corpus_cleaned.json"
    
    post_process_cleaned_corpus(INPUT_FILE, OUTPUT_FILE)