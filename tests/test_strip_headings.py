from arabic_legal_rag.build_corpus import strip_trailing_headings as strip


def test_removes_chapter_heading_with_numbered_subheading():
    text = "وتسقط الدعوى بعد ستة اشهر. الفصل الثالث العمل غير المشروع ١ - المسئولية عن الأعمال الشخصية"
    assert strip(text) == "وتسقط الدعوى بعد ستة اشهر."


def test_removes_book_part_and_chapter_chain():
    text = "ما يرد في هذا الشأن. الباب الأول مصادر الالتزام الفصل الأول العقد ١- الرضاء"
    assert strip(text) == "ما يرد في هذا الشأن."


def test_removes_numbered_subheading_only():
    text = "للمسئول حق الرجوع على الغير. ٣ - المسئولية الناشئة عن الأشياء"
    assert strip(text) == "للمسئول حق الرجوع على الغير."


def test_keeps_body_use_of_fasl_without_ordinal():
    text = "يعتبر الفصل تعسفياً إذا وقع بسبب حجوز، أو وقع هذا الفصل بسبب ديون"
    assert strip(text) == text


def test_keeps_cross_reference_with_range():
    text = "ويراعى في هذه الحالة أحكام المواد من ٢١٥ – إلى ٢٢٠"
    assert strip(text) == text


def test_keeps_heading_word_far_from_the_end():
    body = "ينص الباب الأول على أحكام كثيرة " + ("كلمة " * 120)
    assert strip(body.strip()) == body.strip()