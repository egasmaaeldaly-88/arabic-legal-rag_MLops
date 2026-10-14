from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from arabic_legal_rag.reindex import add_new_articles, indexed_article_numbers


def _doc(num, text="نص"):
    return Document(page_content=f"Article {num}\n[AR]: {text}", metadata={"article_number": num})


def _store(numbers):
    emb = DeterministicFakeEmbedding(size=16)
    return FAISS.from_documents([_doc(n) for n in numbers], emb)


def test_adds_only_new_articles():
    store = _store([1, 2, 3])
    stats = add_new_articles(store, [_doc(3), _doc(4), _doc(5)])
    assert stats == {"before": 3, "added": 2, "skipped": 1, "after": 5}
    assert indexed_article_numbers(store) == {"1", "2", "3", "4", "5"}


def test_new_document_is_searchable():
    store = _store([1, 2, 3])
    add_new_articles(store, [_doc(1150, "مادة تجريبية")])
    hits = store.similarity_search("Article 1150\n[AR]: مادة تجريبية", k=1)
    assert hits[0].metadata["article_number"] == 1150


def test_reindex_is_idempotent_and_batches():
    store = _store([1])
    new = [_doc(n) for n in range(10, 20)]
    first = add_new_articles(store, new, batch_size=3)
    second = add_new_articles(store, new, batch_size=3)
    assert first["added"] == 10 and first["after"] == 11
    assert second["added"] == 0 and second["skipped"] == 10 and second["after"] == 11


def test_duplicates_inside_input_are_added_once():
    store = _store([1])
    stats = add_new_articles(store, [_doc(9), _doc(9)])
    assert stats["added"] == 1 and stats["skipped"] == 1