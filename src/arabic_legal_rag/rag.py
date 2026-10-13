"""Shared RAG helpers: turn retrieved documents into generator context.

Used by both the API and the evaluation script so they always see the same
context for a question.
"""
import re

NOT_FOUND_AR = "لم يتم العثور على مواد قانونية متعلقة."


def is_arabic(text: str) -> bool:
    """True if the text contains Arabic letters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def prepare_context(retriever, question: str):
    """Retrieve documents and build ``(items, sources, arabic)``.

    * items:   ``[{"article": str, "text": str}]`` passed to the generator
    * sources: article citations such as ``["مادة 163", ...]`` (no duplicates)
    * arabic:  whether the question is Arabic (selects text_ar vs text_en)
    """
    arabic = is_arabic(question)
    docs = retriever.invoke(question)

    items, sources = [], []
    for doc in docs:
        art_num = str(doc.metadata.get("article_number", "")).strip()
        if art_num:
            citation = art_num if art_num.startswith("مادة") else f"مادة {art_num}"
            if citation not in sources:
                sources.append(citation)
        text = doc.metadata.get("text_ar" if arabic else "text_en", doc.page_content)
        items.append({"article": art_num, "text": text})
    return items, sources, arabic
