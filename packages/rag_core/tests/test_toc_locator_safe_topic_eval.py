from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from rag_core.rag.toc_locator import TOCIndex
from rag_core.util.query_normalization import normalize_query_text


_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "index"
    / "kankor_gemini_pdf_window2"
    / "toc_manifest.jsonl"
)


@lru_cache(maxsize=1)
def _safe_index() -> TOCIndex:
    return TOCIndex.load(_MANIFEST_PATH, routing_mode="safe_topic_aware")


def _normalized(text: str) -> str:
    return normalize_query_text(text)


def _top_hit(query: str):
    hits = _safe_index().search(question=query, top_k=1)
    return hits[0] if hits else None


POSITIVE_TOPIC_ONLY_CASES = [
    ("توحید در کدام بخش کتاب تعلیمات اسلامی جعفری صنف دهم آمده؟", "G10-Dr-Islamic_Study_jafari", "توحید"),
    ("بخش 2 کتاب تعلیمات اسلامی جعفری صنف 10", "G10-Dr-Islamic_Study_jafari", "توحید"),
    ("مهریه در کتاب تعلیمات اسلامی جعفری صنف 10 کجاست؟", "G10-Dr-Islamic_Study_jafari", "مهریه"),
    ("عزت نفس در جعفری صنف دهم", "G10-Dr-Islamic_Study_jafari", "عزت نفس"),
    ("رشوه در کتاب تعلیمات اسلامی جعفری", "G10-Dr-Islamic_Study_jafari", "رشوه"),
    ("مفهوم و اهمیت علم تفسیر در کتاب تفسیر صنف دهم", "G10-Dr-Tafseer", "مفهوم و اهمیت علم تفسیر"),
    ("درس دهم کتاب تفسیر صنف 10", "G10-Dr-Tafseer", "درس دهم"),
    ("ایمان و عمل صالح در تفسیر صنف دهم", "G10-Dr-Tafseer", "ایمان و عمل صالح"),
    ("حقوق والدین در کتاب تفسیر صنف دهم", "G10-Dr-Tafseer", "حقوق والدین"),
    ("درس پنجم تفسیر صنف دهم", "G10-Dr-Tafseer", "درس پنجم"),
    ("جنگ های صلیبی در کتاب تاریخ صنف یازدهم کجاست؟", "G11-Dr-History", "جنگ های صلیبی"),
    ("رنسانس در اروپا در تاریخ صنف 11", "G11-Dr-History", "رنسانس در اروپا"),
    ("قیام قندهار به رهبری حاجی میرویس نیکه در کتاب تاریخ صنف یازدهم", "G11-Dr-History", "قیام قندهار"),
    ("فصل 2 کتاب تاریخ صنف 11", "G11-Dr-History", "ورود و انتشار اسلام"),
    ("صفوی ها و افغانستان در کتاب تاریخ صنف یازدهم", "G11-Dr-History", "صفوی ها و افغانستان"),
    ("درس اول تفسیر صنف یازدهم", "G11-Dr-Tafseer", "درس اول"),
    ("درس دهم تفسیر صنف یازدهم", "G11-Dr-Tafseer", "درس دهم"),
    ("روزه و فواید آن در کتاب تفسیر 11", "G11-Dr-Tafseer", "روزه و فواید آن"),
    ("صفات بنده گان خاص خداوند در تفسیر صنف یازدهم", "G11-Dr-Tafseer", "صفات بنده گان خاص خداوند"),
    ("حج یا کنگره جهانی مؤمنان در تفسیر صنف 11", "G11-Dr-Tafseer", "حج یا کنگره جهانی"),
    ("calligraphy where is it in grade 12 english book?", "G12-Ps-English", "calligraphy"),
    ("unit 11 in grade 12 english book", "G12-Ps-English", "calligraphy"),
    ("ramadan in grade 12 english book", "G12-Ps-English", "ramadan"),
    ("the wonderful world of the web in grade 12 english", "G12-Ps-English", "wonderful world of the web"),
    ("king ghazi amanullah khan where is it in grade 12 english book", "G12-Ps-English", "king ghazi amanullah khan"),
]

NEGATIVE_CASES = [
    "کتاب تاریخ صنف یازدهم چند صفحه دارد",
    "نویسنده کتاب جعفری کیست",
    "برنامه درسی چیست",
    "page count of grade 12 english book",
    "who published the tafseer book",
    "what is the cover color of grade 10 tafseer",
    "when was grade 11 history printed",
    "isbn of grade 12 english",
    "کتاب جغرافیه صنف دهم",
    "which books are in grade 10",
    "مضمون تاریخ را معرفی کن",
    "در این کتاب چند درس است",
    "what is the textbook size",
    "publisher of grade 10 book",
    "table of contents metadata",
]

AMBIGUOUS_CASES = [
    ("اسلام در افغانستان", {"G11-Dr-History"}),
    ("اروپا در قرون", {"G11-Dr-History"}),
    ("افغانستان", {"G11-Dr-History", "G12-Ps-English"}),
    ("خداوند", {"G10-Dr-Tafseer", "G11-Dr-Tafseer"}),
    ("انسان", {"G10-Dr-Tafseer", "G11-Dr-Tafseer"}),
    ("حقوق", {"G10-Dr-Islamic_Study_jafari", "G10-Dr-Tafseer"}),
    ("توحید", {"G10-Dr-Islamic_Study_jafari"}),
    ("صفات", {"G11-Dr-Tafseer"}),
    ("اروپا", {"G11-Dr-History"}),
    ("افغانستان در قرون", {"G11-Dr-History"}),
    ("مؤمنان", {"G11-Dr-Tafseer"}),
    ("احسان", {"G10-Dr-Tafseer", "G11-Dr-Tafseer"}),
    ("calligraphy", {"G12-Ps-English"}),
    ("web", {"G12-Ps-English"}),
    ("ramadan", {"G12-Ps-English"}),
]

NOISY_CASES = [
    ("توحيد  جعفری  صنف10  ؟", "G10-Dr-Islamic_Study_jafari"),
    ("بخش-2 جعفري 10", "G10-Dr-Islamic_Study_jafari"),
    ("مهريه جعفری 10", "G10-Dr-Islamic_Study_jafari"),
    ("عزت  نفس  جعفري", "G10-Dr-Islamic_Study_jafari"),
    ("رشوه.. جعفری", "G10-Dr-Islamic_Study_jafari"),
    ("درس10 تفسير صنف10", "G10-Dr-Tafseer"),
    ("ايمان و عمل صالح تفسير10", "G10-Dr-Tafseer"),
    ("حقوق والدين تفسير 10", "G10-Dr-Tafseer"),
    ("درس 5 تفسير10", "G10-Dr-Tafseer"),
    ("مفهوم اهمیت علم تفسير", "G10-Dr-Tafseer"),
    ("جنگ هاي صليبي تاريخ 11", "G11-Dr-History"),
    ("رنسانس   اروپا 11", "G11-Dr-History"),
    ("قيام قندهار تاريخ11", "G11-Dr-History"),
    ("فصل 2 تاريخ 11", "G11-Dr-History"),
    ("صفوي ها افغانستان تاريخ11", "G11-Dr-History"),
    ("درس1 تفسير صنف11", "G11-Dr-Tafseer"),
    ("درس10 تفسير صنف11", "G11-Dr-Tafseer"),
    ("روزه فواید آن تفسير11", "G11-Dr-Tafseer"),
    ("صفات بنده گان خاص خداوند تفسير11", "G11-Dr-Tafseer"),
    ("حج کنگره جهانی مؤمنان تفسير11", "G11-Dr-Tafseer"),
    ("CALLIGRAPHY grade12 english", "G12-Ps-English"),
    ("unit11 grade12 english", "G12-Ps-English"),
    ("ramadan grade12 eng book", "G12-Ps-English"),
    ("wonderful world web grade12 english", "G12-Ps-English"),
    ("king ghazi amanullah khan grade12 english", "G12-Ps-English"),
]


@pytest.mark.parametrize(("query", "expected_source_id", "expected_title_fragment"), POSITIVE_TOPIC_ONLY_CASES)
def test_safe_topic_eval_positive_cases(query: str, expected_source_id: str, expected_title_fragment: str) -> None:
    hit = _top_hit(query)
    assert hit is not None
    metadata = hit.document.metadata
    assert metadata.get("source_id") == expected_source_id
    assert _normalized(expected_title_fragment) in _normalized(str(metadata.get("chapter_title", "")))


@pytest.mark.parametrize("query", NEGATIVE_CASES)
def test_safe_topic_eval_negative_cases(query: str) -> None:
    assert _safe_index().search(question=query, top_k=1) == []


@pytest.mark.parametrize(("query", "allowed_sources"), AMBIGUOUS_CASES)
def test_safe_topic_eval_ambiguous_cases_do_not_route_to_wrong_book(
    query: str,
    allowed_sources: set[str],
) -> None:
    hit = _top_hit(query)
    if hit is None:
        return
    assert hit.document.metadata.get("source_id") in allowed_sources


@pytest.mark.parametrize(("query", "expected_source_id"), NOISY_CASES)
def test_safe_topic_eval_noisy_cases(query: str, expected_source_id: str) -> None:
    hit = _top_hit(query)
    assert hit is not None
    assert hit.document.metadata.get("source_id") == expected_source_id
