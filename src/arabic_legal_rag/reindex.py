"""Batch re-indexing: add NEW articles to the existing FAISS index without rebuilding it.

Only the new articles are embedded (bge-m3), so this takes seconds instead of re-embedding
the whole corpus. Articles whose number is already indexed are skipped (never duplicated).

Usage (from the repo root):

    python -m arabic_legal_rag.reindex --input data/new_articles_sample.json

Input file: a JSON list in the same schema as data/legal_corpus_structured.json
(article_number, text_ar, text_en, book, chapter, section, topic, ...).

After re-indexing, restart the API / BentoML service so it loads the updated index.
Note: dvc repro build_index rebuilds the index from data/legal_corpus_structured.json, so
to keep a new article permanently, also add it to that corpus file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def indexed_article_numbers(vector_store) -> set[str]:
    """Article numbers already present in the store (as strings, so '6' == 6)."""
    return {
        str(doc.metadata.get("article_number"))
        for doc in vector_store.docstore._dict.values()
    }


def add_new_articles(vector_store, docs: list, batch_size: int = 32) -> dict:
    """Add the documents whose article_number is not yet indexed.

    Returns {"before": int, "added": int, "skipped": int, "after": int}.
    """
    before = int(vector_store.index.ntotal)
    seen = indexed_article_numbers(vector_store)

    new_docs = []
    skipped = 0
    for doc in docs:
        num = str(doc.metadata.get("article_number"))
        if num in seen:
            skipped += 1
            continue
        seen.add(num)  # also guards against duplicates inside the input file
        new_docs.append(doc)

    for start in range(0, len(new_docs), batch_size):
        vector_store.add_documents(new_docs[start:start + batch_size])

    return {
        "before": before,
        "added": len(new_docs),
        "skipped": skipped,
        "after": int(vector_store.index.ntotal),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, help="JSON file with the new articles")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)

    # Heavy imports live here so the core logic above stays easy to unit-test.
    from arabic_legal_rag.ingest import process_corpus
    from arabic_legal_rag.model import get_embedding_model, load_vector_store
    from arabic_legal_rag.utils import load_config

    config = load_config(args.config)
    index_dir = Path(config["vector_db"]["index_dir"])
    if not index_dir.exists():
        print(f"Index not found at {index_dir}. Run `dvc repro build_index` first.", file=sys.stderr)
        return 1

    docs = process_corpus(args.input)  # same filtering/format as the full ingestion
    if not docs:
        print("No valid articles in the input file.", file=sys.stderr)
        return 1

    embeddings = get_embedding_model(config["model"]["embedding_model"])
    store = load_vector_store(str(index_dir), embeddings)

    stats = add_new_articles(store, docs, batch_size=args.batch_size)
    if stats["added"]:
        store.save_local(str(index_dir.resolve()))

    print(
        f"documents_indexed: {stats['before']} -> {stats['after']} "
        f"(added {stats['added']}, skipped {stats['skipped']} already indexed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())