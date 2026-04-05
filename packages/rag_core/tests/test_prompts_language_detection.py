from __future__ import annotations

from rag_core.rag.prompts import detect_answer_language


def test_detect_answer_language_match_user_pashto() -> None:
    assert detect_answer_language("سلام! څنګه مرسته درسره وکړم؟", default_language="match-user") == "پښتو"


def test_detect_answer_language_match_user_english_defaults_to_dari() -> None:
    assert detect_answer_language("Explain Newton's second law.", default_language="match-user") == "دری"


def test_detect_answer_language_match_user_arabic_defaults_to_dari() -> None:
    assert detect_answer_language("اشرح قانون نيوتن الثاني مع مثال بسيط.", default_language="match-user") == "دری"


def test_detect_answer_language_match_user_defaults_to_dari() -> None:
    assert detect_answer_language("سلام، قانون دوم نیوتن چیست؟", default_language="match-user") == "دری"


def test_detect_answer_language_explicit_pashto_stays_pashto() -> None:
    assert detect_answer_language("Explain Newton's second law.", default_language="ps") == "پښتو"
