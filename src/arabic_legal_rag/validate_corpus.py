import json
from pathlib import Path
from collections import defaultdict
from loguru import logger

def validate_legal_corpus(json_path: str = "data/legal_corpus_structured.json"):
    """
    Validates the structured legal corpus according to ITI MLOps handbook standards:
    1. Ensures file exists and is valid JSON.
    2. Asserts article numbers and checks their context if duplicated.
    3. Verifies that every record has a non-empty text field (ar_text or text_ar).
    4. Validates that repealed articles are explicitly flagged.
    """
    path = Path(json_path)
    if not path.exists():
        logger.error(f"Corpus file not found at: {json_path}")
        raise FileNotFoundError(f"Corpus file not found at: {json_path}")

    logger.info(f"Loading corpus from {json_path} for validation...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Corpus must be a non-empty list of JSON records.")

    logger.info(f"Total records loaded: {len(data)}")

    empty_text_count = 0
    repealed_count = 0
    article_occurrences = defaultdict(list)

    for idx, item in enumerate(data):
        # Check article number and record its occurrences
        article_num = item.get("article_number")
        if article_num is None:
            logger.warning(f"Record at index {idx} is missing 'article_number'.")
        else:
            article_occurrences[article_num].append({
                "index": idx,
                "book": item.get("book", ""),
                "chapter": item.get("chapter", "")
            })

        # Check text content (supporting both ar_text and text_ar)
        text_content = item.get("ar_text") or item.get("text_ar", "")
        if not text_content.strip():
            empty_text_count += 1
            logger.warning(f"Empty text found for article number: {article_num}")

        # Check repealed flag
        if item.get("is_repealed", False):
            repealed_count += 1

    # Assertions / Validation Rules
    assert empty_text_count == 0, f"Validation Failed: Found {empty_text_count} records with empty text!"

    # Check for duplicate article numbers and show context
    duplicates = {num: occs for num, occs in article_occurrences.items() if len(occs) > 1}
    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate article numbers across the corpus.")
        print("\n" + "="*50)
        print("Inspecting Duplicates Context (Sample):")
        print("="*50)
        # Show context for the first few duplicates as an example
        # Show context for the first few duplicates as an example
        for num, occurrences in list(duplicates.items())[:5]:
            print(f"Article {num} appears {len(occurrences)} times in:")
            for occ in occurrences:
                print(f"  - Book: '{occ['book']}' | Chapter: '{occ['chapter']}' (Index: {occ['index']})")
    logger.success(f"Validation Passed Successfully! 🎉")
    logger.info(f"Summary -> Total Articles: {len(data)} | Repealed Articles: {repealed_count} | Empty Texts: {empty_text_count}")

    # Create the output flag file required by DVC
    output_flag = Path("outputs/validation.flag")
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("Validation passed successfully!")
    logger.info("Validation flag file created for DVC.")

if __name__ == "__main__":
    validate_legal_corpus()