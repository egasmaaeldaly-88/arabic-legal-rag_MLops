from arabic_legal_rag.build_corpus import (
    clean_arabic_line,
    find_repealed_ranges,
    fix_ligature_chars,
    format_missing,
    split_article_blocks,
    split_page,
)


def test_fix_ligature_chars_reverses_only_ligature_glyphs():
    chars = [{"text": "لا"}, {"text": "لإ"}, {"text": "ر"}, {"text": "لأ"}, {"text": "لآ"}]
    assert fix_ligature_chars(chars) == 4
    assert [c["text"] for c in chars] == ["ال", "إل", "ر", "أل", "آل"]


def test_ligature_survives_token_reversal():
    # raw visual token for "الإصدار" after the glyph fix: ر ا د ص [إل] ا
    assert clean_arabic_line("رادصإلا") == "الإصدار"
    # raw visual token for "الانتفاع": ع ا ف ت ن [ال] ا
    assert clean_arabic_line("عافتنالا") == "الانتفاع"
    # standalone "لا" (negation) is one ligature glyph, pre-reversed to "ال"
    assert clean_arabic_line("ال") == "لا"


def test_digit_runs_are_not_reversed():
    assert clean_arabic_line("نوناقلاو١٨٨٣") == "١٨٨٣والقانون"
    assert clean_arabic_line("١٩٤٩") == "١٩٤٩"
    assert clean_arabic_line("ربوتكأ ٢٨") == "٢٨ أكتوبر"


def test_page_one_keeps_only_the_code_not_the_promulgation_law():
    text = "\n".join(
        ["نوناق", "١ ةدام", "قديم", "٢ ةدام", "ثان قديم",
         "١ ةدام", "جديد", "٢ ةدام", "ثان جديد", "٣ ةدام", "ثالث"]
    )
    blocks = split_article_blocks(text, page_idx=0)
    assert [n for n, _ in blocks] == ["١", "٢", "٣"]
    assert blocks[0][1] == "جديد"
    # later pages are not trimmed
    assert len(split_article_blocks(text, page_idx=3)) == 5


def test_cross_reference_is_not_treated_as_a_heading():
    text = "٥٥٨ ةدام\nوفقا ل ٥٦٠ ةداملا نهاية النص\n٥٥٩ ةدام\nآخر"
    blocks = split_article_blocks(text, page_idx=5)
    assert [n for n, _ in blocks] == ["٥٥٨", "٥٥٩"]
    assert "٥٦٠ ةداملا" in blocks[0][1]


def test_format_missing_compacts_ranges():
    assert format_missing([1, 2, 3, 7], last=10) == "6 missing: 4-6, 8-10"
    assert format_missing(range(1, 6), last=5) == "0 missing: none"


def test_split_page_returns_leading_text_of_the_continued_article():
    text = "تتمة المادة السابقة\n٧ ةدام\nنص\n٨ ةدام\nنص آخر"
    leading, blocks = split_page(text, page_idx=3)
    assert leading.strip() == "تتمة المادة السابقة"
    assert [n for n, _ in blocks] == ["٧", "٨"]


def test_find_repealed_ranges_from_english_and_arabic():
    en = "Article 54 * Articles 54-80 have been\nrepealed by Presidential Decree."
    ar = "ةاغلم ٤١٧ ىلإ ٣٨٩ نم داوملا"
    assert find_repealed_ranges(en, ar) == {(54, 80), (389, 417)}
    assert find_repealed_ranges("Article 5 is in force.", "") == set()
