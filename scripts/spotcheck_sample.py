"""Pick 20 random articles for the manual extraction spot-check (handbook: 20 articles).

Usage (from the repo root):
    python scripts/spotcheck_sample.py

Writes reports/extraction_spotcheck.md: one row per sampled article with its PDF page and the first
characters of the extracted Arabic text. Open the PDF at that page, compare the text, and fill the
"Result" column (OK, or what is wrong). The seed is fixed, so everyone gets the same 20 articles.
"""
import json
import random
from pathlib import Path

CORPUS = Path("data/legal_corpus_structured.json")
OUT = Path("reports/extraction_spotcheck.md")
SEED, N = 42, 20


def main() -> None:
    articles = json.loads(CORPUS.read_text(encoding="utf-8"))
    repealed = [a for a in articles if str(a.get("is_repealed")).lower() == "true"]
    others = [a for a in articles if a not in repealed]
    rng = random.Random(SEED)
    # keep up to 2 repealed articles in the sample so the repealed flag gets checked too
    picked = rng.sample(repealed, min(2, len(repealed)))
    picked += rng.sample(others, N - len(picked))
    picked.sort(key=lambda a: int(a["article_number"]))

    lines = [
        f"# Extraction spot-check ({N} articles, seed {SEED})",
        "",
        "Compare each extracted text with the PDF page. Result: OK, or describe the problem.",
        "",
        "| Article | PDF page | Repealed flag | Extracted text (start) | Result |",
        "| :---: | :---: | :---: | :--- | :--- |",
    ]
    for a in picked:
        start = " ".join(a["text_ar"].split())[:90].replace("|", "/")
        lines.append(
            f"| {a['article_number']} | {a.get('source_page', '')} | {a.get('is_repealed')} | {start} |  |"
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}: articles {[a['article_number'] for a in picked]}")


if __name__ == "__main__":
    main()
