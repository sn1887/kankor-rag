from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "adaptive_attachment_eval.py"
    spec = importlib.util.spec_from_file_location("adaptive_attachment_eval", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def eval_module():
    return _load_script_module()


def _config(module):
    return module.EvalConfig(
        max_attachments=3,
        min_attachments=1,
        top_score_low=0.42,
        top_score_very_low=0.30,
        score_gap_low=0.05,
        complexity_length_tokens=25,
    )


def _suite_row(module, **kwargs):
    return module.SuiteRow.model_validate(kwargs)


def _retrieval_row(module, **kwargs):
    return module.RetrievalRow.model_validate(kwargs)


def test_load_suite_rows_rejects_invalid_row(tmp_path: Path, eval_module) -> None:
    suite_path = tmp_path / "suite.jsonl"
    _write_jsonl(
        suite_path,
        [
            {
                "id": "A1",
                "category": "A",
                "intent": "grounded_textbook",
                "expected_count": 1,
            }
        ],
    )

    with pytest.raises(SystemExit, match="Invalid suite row"):
        eval_module.load_suite_rows(suite_path)


def test_load_suite_rows_rejects_duplicate_ids(tmp_path: Path, eval_module) -> None:
    suite_path = tmp_path / "suite.jsonl"
    _write_jsonl(
        suite_path,
        [
            {
                "id": "A1",
                "category": "A",
                "intent": "grounded_textbook",
                "query": "سوال اول",
                "expected_count": 1,
            },
            {
                "id": "A1",
                "category": "A",
                "intent": "grounded_textbook",
                "query": "سوال دوم",
                "expected_count": 1,
            },
        ],
    )

    with pytest.raises(SystemExit, match="Duplicate suite id"):
        eval_module.load_suite_rows(suite_path)


@pytest.mark.parametrize(
    "hit_payload",
    [
        {"source_id": "book-a"},
        {"score": 0.41},
    ],
)
def test_load_retrieval_rows_rejects_missing_required_hit_fields(
    tmp_path: Path,
    eval_module,
    hit_payload: dict,
) -> None:
    retrieval_path = tmp_path / "retrieval_results.jsonl"
    _write_jsonl(
        retrieval_path,
        [
            {
                "query_id": "A1",
                "intent": "grounded_textbook",
                "query": "سوال",
                "hits": [hit_payload],
            }
        ],
    )

    with pytest.raises(SystemExit, match="Invalid retrieval row"):
        eval_module.load_retrieval_rows(retrieval_path)


def test_evaluate_row_exact_expected_pass_and_fail(eval_module) -> None:
    cfg = _config(eval_module)
    suite = _suite_row(
        eval_module,
        id="A1",
        category="A",
        intent="grounded_textbook",
        query="سوال",
        expected_count=1,
    )
    retrieval_pass = _retrieval_row(
        eval_module,
        query_id="A1",
        query="سوال",
        intent="grounded_textbook",
        hits=[
            {"score": 0.91, "source_id": "book-a"},
            {"score": 0.70, "source_id": "book-a"},
        ],
    )
    row_pass = eval_module.evaluate_row(suite_row=suite, retrieval_row=retrieval_pass, config=cfg)
    assert row_pass["adaptive_attachment_count"] == 1
    assert row_pass["status"] == "pass"

    retrieval_fail = _retrieval_row(
        eval_module,
        query_id="A1",
        query="سوال",
        intent="grounded_textbook",
        hits=[
            {"score": 0.41, "source_id": "book-a"},
            {"score": 0.40, "source_id": "book-b"},
        ],
    )
    row_fail = eval_module.evaluate_row(suite_row=suite, retrieval_row=retrieval_fail, config=cfg)
    assert row_fail["adaptive_attachment_count"] == 2
    assert row_fail["status"] == "fail"


def test_evaluate_row_range_expected_pass_and_fail(eval_module) -> None:
    cfg = _config(eval_module)
    suite = _suite_row(
        eval_module,
        id="E3",
        category="E",
        intent="grounded_textbook",
        query="تاریخ افغانستان",
        expected_count={"min": 2, "max": 3},
    )
    retrieval_pass = _retrieval_row(
        eval_module,
        query_id="E3",
        query="تاریخ افغانستان",
        intent="grounded_textbook",
        hits=[
            {"score": 0.41, "source_id": "book-a"},
            {"score": 0.40, "source_id": "book-b"},
        ],
    )
    row_pass = eval_module.evaluate_row(suite_row=suite, retrieval_row=retrieval_pass, config=cfg)
    assert row_pass["adaptive_attachment_count"] == 2
    assert row_pass["status"] == "pass"

    retrieval_fail = _retrieval_row(
        eval_module,
        query_id="E3",
        query="تاریخ افغانستان",
        intent="grounded_textbook",
        hits=[
            {"score": 0.90, "source_id": "book-a"},
            {"score": 0.80, "source_id": "book-a"},
        ],
    )
    row_fail = eval_module.evaluate_row(suite_row=suite, retrieval_row=retrieval_fail, config=cfg)
    assert row_fail["adaptive_attachment_count"] == 1
    assert row_fail["status"] == "fail"


def test_evaluate_row_zero_hit_allowance_and_skip(eval_module) -> None:
    cfg = _config(eval_module)
    suite_allow = _suite_row(
        eval_module,
        id="E1",
        category="E",
        intent="grounded_textbook",
        query="سوال بیرون از کورپس",
        expected_count={"min": 0, "max": 1},
        allow_zero_hits=True,
    )
    retrieval_empty = _retrieval_row(
        eval_module,
        query_id="E1",
        query="سوال بیرون از کورپس",
        intent="grounded_textbook",
        hits=[],
    )
    row_allow = eval_module.evaluate_row(suite_row=suite_allow, retrieval_row=retrieval_empty, config=cfg)
    assert row_allow["adaptive_attachment_count"] == 0
    assert row_allow["status"] == "pass"

    suite_skip = _suite_row(
        eval_module,
        id="A2",
        category="A",
        intent="grounded_textbook",
        query="سوال",
        expected_count=1,
    )
    row_skip = eval_module.evaluate_row(suite_row=suite_skip, retrieval_row=retrieval_empty, config=cfg)
    assert row_skip["status"] == "skipped_no_hits"
    assert row_skip["failure_class"] == "skipped_no_hits"


def test_category_d_failure_class_disambiguates_retrieval_spread(eval_module) -> None:
    cfg = _config(eval_module)
    suite = _suite_row(
        eval_module,
        id="D1",
        category="D",
        intent="grounded_textbook",
        query="سوال",
        expected_count=1,
    )
    retrieval_spread = _retrieval_row(
        eval_module,
        query_id="D1",
        query="سوال",
        intent="grounded_textbook",
        hits=[
            {"score": 0.50, "source_id": "book-a"},
            {"score": 0.49, "source_id": "book-b"},
        ],
    )
    row_spread = eval_module.evaluate_row(suite_row=suite, retrieval_row=retrieval_spread, config=cfg)
    assert row_spread["status"] == "fail"
    assert row_spread["failure_class"] == "retrieval_spread"

    retrieval_single = _retrieval_row(
        eval_module,
        query_id="D1",
        query="سوال",
        intent="grounded_textbook",
        hits=[
            {"score": 0.41, "source_id": "book-a"},
            {"score": 0.39, "source_id": "book-a"},
        ],
    )
    row_single = eval_module.evaluate_row(suite_row=suite, retrieval_row=retrieval_single, config=cfg)
    assert row_single["status"] == "fail"
    assert row_single["failure_class"] == "policy_mismatch"


def test_build_summary_includes_confusion_matrix_and_pass_rate_denominators(eval_module) -> None:
    cfg = _config(eval_module)
    rows = [
        {
            "query_id": "A1",
            "category": "A",
            "status": "pass",
            "failure_class": "pass",
            "expected_label": "1",
            "actual_label": "1",
            "distance_from_expected_midpoint": 0.0,
            "top_score": 0.9,
            "reason_flags": [],
        },
        {
            "query_id": "A2",
            "category": "A",
            "status": "fail",
            "failure_class": "policy_mismatch",
            "expected_label": "1",
            "actual_label": "2",
            "distance_from_expected_midpoint": 1.0,
            "top_score": 0.6,
            "reason_flags": ["low_top_score"],
        },
        {
            "query_id": "D1",
            "category": "D",
            "status": "fail",
            "failure_class": "retrieval_spread",
            "expected_label": "1",
            "actual_label": "2",
            "distance_from_expected_midpoint": 1.0,
            "top_score": 0.5,
            "reason_flags": ["multi_source_top3"],
        },
        {
            "query_id": "E1",
            "category": "E",
            "status": "skipped_no_hits",
            "failure_class": "skipped_no_hits",
            "expected_label": "0-1",
            "actual_label": "skipped",
            "distance_from_expected_midpoint": None,
            "top_score": None,
            "reason_flags": [],
        },
    ]
    summary = eval_module.build_summary(rows=rows, suite_version="v1", config=cfg)

    assert summary["suite_version"] == "v1"
    assert summary["rows_total"] == 4
    assert summary["rows_non_skipped"] == 3
    assert summary["rows_skipped_no_hits"] == 1
    assert summary["strict_pass_rate"] == pytest.approx(1 / 3)
    assert summary["policy_pass_rate"] == pytest.approx(1 / 2)
    assert set(summary["confusion_matrix"].keys()) >= {"1", "2", "3", "2-3", "0-1"}
    assert summary["confusion_matrix"]["1"]["1"] == 1
    assert summary["confusion_matrix"]["1"]["2"] == 2
    assert summary["confusion_matrix"]["0-1"]["skipped"] == 1
    assert "top_score_low" in summary["threshold_snapshot"]


def test_build_summary_hardest_rows_sort_by_distance_then_top_score(eval_module) -> None:
    cfg = _config(eval_module)
    rows = [
        {
            "query_id": "q-high-gap",
            "category": "A",
            "status": "fail",
            "failure_class": "policy_mismatch",
            "expected_label": "1",
            "actual_label": "3",
            "distance_from_expected_midpoint": 2.0,
            "top_score": 0.8,
            "reason_flags": [],
        },
        {
            "query_id": "q-low-score",
            "category": "A",
            "status": "fail",
            "failure_class": "policy_mismatch",
            "expected_label": "1",
            "actual_label": "2",
            "distance_from_expected_midpoint": 1.0,
            "top_score": 0.2,
            "reason_flags": [],
        },
        {
            "query_id": "q-high-score",
            "category": "A",
            "status": "fail",
            "failure_class": "policy_mismatch",
            "expected_label": "1",
            "actual_label": "2",
            "distance_from_expected_midpoint": 1.0,
            "top_score": 0.7,
            "reason_flags": [],
        },
    ]
    summary = eval_module.build_summary(rows=rows, suite_version="v1", config=cfg)
    hardest = summary["hardest_rows_top10"]

    assert hardest[0]["query_id"] == "q-high-gap"
    assert hardest[1]["query_id"] == "q-low-score"
    assert hardest[2]["query_id"] == "q-high-score"


def test_run_eval_writes_outputs_and_infers_suite_version(tmp_path: Path, eval_module) -> None:
    suite_path = tmp_path / "adaptive_attachment_threshold_suite_v1.jsonl"
    retrieval_path = tmp_path / "retrieval_results.jsonl"
    output_dir = tmp_path / "out"

    _write_jsonl(
        suite_path,
        [
            {
                "id": "A1",
                "category": "A",
                "intent": "grounded_textbook",
                "query": "قانون اول نیوتن چیست؟",
                "expected_count": 1,
            }
        ],
    )
    _write_jsonl(
        retrieval_path,
        [
            {
                "query_id": "A1",
                "intent": "grounded_textbook",
                "query": "قانون اول نیوتن چیست؟",
                "hits": [
                    {"score": 0.91, "source_id": "book-a"},
                    {"score": 0.70, "source_id": "book-a"},
                ],
            }
        ],
    )

    results_path, summary_path = eval_module.run_eval(
        suite_path=suite_path,
        retrieval_results_path=retrieval_path,
        output_dir=output_dir,
        suite_version=None,
        config=_config(eval_module),
    )

    assert results_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["suite_version"] == "v1"
