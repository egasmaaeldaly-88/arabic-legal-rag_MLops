import json

from arabic_legal_rag.ingest import process_corpus


def test_process_corpus_skips_repealed_placeholders(tmp_path):
    records = [
        {"article_number": 1, "text_ar": "نص", "text_en": "text"},
        {"article_number": 60, "text_ar": "مادة ملغاة", "text_en": "repealed",
         "is_repealed": True, "is_stub": True},
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    docs = process_corpus(str(path))
    assert [d.metadata["article_number"] for d in docs] == [1]
