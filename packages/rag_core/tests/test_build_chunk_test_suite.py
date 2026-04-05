from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "build_chunk_test_suite.py"
    spec = importlib.util.spec_from_file_location("build_chunk_test_suite", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lookup_chunk_test_spec_handles_known_file_names() -> None:
    module = _load_module()
    root = Path("data/Chunk_test/questions")
    spec = module.lookup_chunk_test_spec(root / "grade 10" / "cpmputer.json", root)
    assert spec.source_id == "G10-Dr-Computer"
    assert spec.subject == "computer_science"

    spec_grade_12 = module.lookup_chunk_test_spec(root / "grade 12" / "islamicHanafi.json", root)
    assert spec_grade_12.source_id == "G12-Dr-Islamic_Study_Hanafi"


def test_iter_language_variants_uses_english_fallback_fields() -> None:
    module = _load_module()
    row = {
        "question_dari": "سوال دری",
        "question_pashto": "پوښتنه پښتو",
        "english_question": "English fallback",
    }
    variants = module.iter_language_variants(row)
    assert variants == [
        ("fa", "سوال دری", "question_dari"),
        ("ps", "پوښتنه پښتو", "question_pashto"),
        ("en", "English fallback", "english_question"),
    ]


def test_classify_source_row_filters_frontmatter_metadata_queries() -> None:
    module = _load_module()
    row = {
        "question_dari": "در صفحهٔ مشخصات کتاب، مضمون این کتاب چه نوشته شده است؟",
        "question_pashto": "د کتاب د ځانګړنو په پاڼه کې د دې کتاب مضمون څه لیکل شوی دی؟",
        "question_english": "On the book information page, what subject is this textbook identified as?",
        "pdf_page_start": 4,
        "pdf_page_end": 4,
        "final_check": "ok",
    }
    included, reason = module.classify_source_row(row)
    assert included is False
    assert reason == "frontmatter_metadata_query"


def test_build_chunk_test_rows_expands_multilingual_rows_and_qrels(tmp_path) -> None:
    module = _load_module()
    data_dir = tmp_path / "grade 12"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "id": "Q014",
            "question_dari": "سوال دری",
            "question_pashto": "پوښتنه پښتو",
            "question_english": "English question",
            "correct_option": "A",
            "options": {"A": "جواب"},
            "answer_text": "جواب",
            "pdf_page_start": 12,
            "pdf_page_end": 13,
            "section_title": "درس",
            "evidence_snippet": "شاهد",
            "validation_note": "ok",
            "final_check": "ok",
            "reasoning_type": "Fact_Lookup",
            "is_multihop": False,
        }
    ]
    (data_dir / "Biology.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    suite_rows, qrel_rows, summary = module.build_chunk_test_rows(chunk_test_root=tmp_path)

    assert [row["id"] for row in suite_rows] == [
        "chunk_v2_g12_dr_biology_q014_fa",
        "chunk_v2_g12_dr_biology_q014_ps",
        "chunk_v2_g12_dr_biology_q014_en",
    ]
    assert [row["query_id"] for row in qrel_rows] == [
        "chunk_v2_g12_dr_biology_q014_fa",
        "chunk_v2_g12_dr_biology_q014_ps",
        "chunk_v2_g12_dr_biology_q014_en",
    ]
    assert summary["language_counts"] == {"en": 1, "fa": 1, "ps": 1}
    assert summary["rows_total"] == 3
    assert summary["qrels_total"] == 3


def test_build_chunk_test_rows_excludes_invalid_rows_and_tracks_reasons(tmp_path) -> None:
    module = _load_module()
    data_dir = tmp_path / "grade 11"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "id": "Q001",
            "question_dari": "در صفحه مشخصات کتاب، مضمون این کتاب چه آمده است؟",
            "question_pashto": "د کتاب د ځانګړنو په پاڼه کې د دې کتاب مضمون څه راغلی دی؟",
            "question_english": "On the book information page, what subject is this textbook identified as?",
            "answer_text": "جواب",
            "pdf_page_start": 4,
            "pdf_page_end": 4,
            "final_check": "ok",
        },
        {
            "id": "Q002",
            "question_dari": "سوال معتبر",
            "question_pashto": "پوښتنه معتبره",
            "answer_text": "جواب",
            "pdf_page_start": "",
            "pdf_page_end": 9,
            "final_check": "ok",
        },
        {
            "id": "Q003",
            "question_dari": "سوال باقی‌مانده",
            "question_pashto": "وروستۍ پوښتنه",
            "english": "English fallback text",
            "answer_text": "جواب",
            "pdf_page_start": 8,
            "pdf_page_end": 9,
            "final_check": "pdf page verified",
        },
    ]
    (data_dir / "history.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    suite_rows, qrel_rows, summary = module.build_chunk_test_rows(chunk_test_root=tmp_path)

    assert len(suite_rows) == 3
    assert len(qrel_rows) == 3
    assert summary["filtered_reason_counts"]["frontmatter_metadata_query"] == 1
    assert summary["filtered_reason_counts"]["missing_pdf_page_range"] == 1
    assert summary["source_rows_included"] == 1


def test_build_chunk_test_rows_disambiguates_duplicate_source_row_ids(tmp_path) -> None:
    module = _load_module()
    data_dir = tmp_path / "grade 10"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "id": "Q001",
            "question_dari": "سوال اول",
            "question_pashto": "لومړۍ پوښتنه",
            "question_english": "Question one",
            "answer_text": "جواب",
            "pdf_page_start": 10,
            "pdf_page_end": 10,
            "final_check": "ok",
        },
        {
            "id": "Q001",
            "question_dari": "سوال دوم",
            "question_pashto": "دوهمه پوښتنه",
            "question_english": "Question two",
            "answer_text": "جواب",
            "pdf_page_start": 11,
            "pdf_page_end": 11,
            "final_check": "ok",
        },
    ]
    (data_dir / "history.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    suite_rows, qrel_rows, _summary = module.build_chunk_test_rows(chunk_test_root=tmp_path)

    ids = [row["id"] for row in suite_rows]
    assert "chunk_v2_g10_dr_history_q001_fa" in ids
    assert "chunk_v2_g10_dr_history_q001_dup2_fa" in ids
    assert len(ids) == len(set(ids))
    qrel_ids = [row["query_id"] for row in qrel_rows]
    assert len(qrel_ids) == len(set(qrel_ids))
