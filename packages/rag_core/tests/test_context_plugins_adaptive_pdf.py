from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from rag_core.rag.context_plugins import PdfWindowGroundingContextPlugin
from rag_core.rag.context_plugins import AdaptiveAttachmentRetrievalSignals
from rag_core.rag.context_plugins import select_adaptive_pdf_attachment_count
from rag_core.types import Document, Hit


def _hit(score: float, source_id: str) -> Hit:
    return Hit(
        document=Document(
            id=f"{source_id}:{score}",
            text="",
            metadata={"source_id": source_id},
        ),
        score=score,
    )


def _decide(
    *,
    question: str,
    hits: list[Hit],
    effective_max: int,
    adaptive_enabled: bool = True,
    adaptive_applicable: bool = True,
    min_attachments: int = 1,
    max_attachments: int = 3,
):
    return select_adaptive_pdf_attachment_count(
        question=question,
        hits=hits,
        effective_max=effective_max,
        max_attachments=max_attachments,
        min_attachments=min_attachments,
        adaptive_enabled=adaptive_enabled,
        adaptive_applicable=adaptive_applicable,
        top_score_low=0.42,
        top_score_very_low=0.30,
        score_gap_low=0.05,
        complexity_length_tokens=25,
    )


def test_adaptive_high_confidence_single_source_selects_one() -> None:
    hits = [_hit(0.95, "book-a"), _hit(0.70, "book-a"), _hit(0.62, "book-a")]
    decision = _decide(question="Explain photosynthesis", hits=hits, effective_max=3)
    assert decision.count == 1


def test_adaptive_ambiguous_retrieval_selects_two() -> None:
    hits = [_hit(0.41, "book-a"), _hit(0.38, "book-b"), _hit(0.33, "book-b")]
    decision = _decide(question="What is energy?", hits=hits, effective_max=3)
    assert decision.count == 2
    assert "low_top_score" in decision.reason_flags


def test_adaptive_very_low_top_score_selects_three() -> None:
    hits = [_hit(0.29, "book-a"), _hit(0.28, "book-b"), _hit(0.26, "book-c")]
    decision = _decide(question="Explain this topic", hits=hits, effective_max=3)
    assert decision.count == 3
    assert "very_low_top_score" in decision.reason_flags


def test_adaptive_clamped_by_effective_max_sets_reason_flag() -> None:
    hits = [_hit(0.29, "book-a"), _hit(0.28, "book-b"), _hit(0.26, "book-c")]
    decision = _decide(question="Explain this topic", hits=hits, effective_max=2)
    assert decision.count == 2
    assert "clamped_by_effective_max" in decision.reason_flags


def test_adaptive_complex_query_with_source_spread_selects_three() -> None:
    hits = [_hit(0.82, "book-a"), _hit(0.70, "book-b"), _hit(0.60, "book-b")]
    decision = _decide(question="مقایسه فصل اول و دوم را هم بگو", hits=hits, effective_max=3)
    assert decision.count == 3
    assert "complex_query_with_source_spread" in decision.reason_flags


def test_adaptive_question_complexity_alone_does_not_escalate() -> None:
    hits = [_hit(0.90, "book-a"), _hit(0.75, "book-a"), _hit(0.70, "book-a")]
    decision = _decide(question="مقایسه این موضوع را بگو", hits=hits, effective_max=3)
    assert decision.count == 1


def test_boundary_top_score_equal_threshold_does_not_escalate() -> None:
    hits = [_hit(0.42, "book-a"), _hit(0.20, "book-a"), _hit(0.10, "book-a")]
    decision = _decide(question="simple question", hits=hits, effective_max=3)
    assert decision.count == 1


def test_boundary_top_score_just_below_threshold_escalates() -> None:
    hits = [_hit(0.4199, "book-a"), _hit(0.20, "book-a"), _hit(0.10, "book-a")]
    decision = _decide(question="simple question", hits=hits, effective_max=3)
    assert decision.count == 2


def test_boundary_score_gap_equal_threshold_does_not_escalate() -> None:
    hits = [_hit(0.90, "book-a"), _hit(0.85, "book-a"), _hit(0.70, "book-a")]
    decision = _decide(question="simple question", hits=hits, effective_max=3)
    assert decision.count == 1


def test_boundary_score_gap_just_below_threshold_escalates() -> None:
    hits = [_hit(0.90, "book-a"), _hit(0.8501, "book-a"), _hit(0.70, "book-a")]
    decision = _decide(question="simple question", hits=hits, effective_max=3)
    assert decision.count == 2


def test_selector_score_gap_override_supersedes_post_expansion_gap() -> None:
    hits = [_hit(0.90, "book-a"), _hit(0.899, "book-a"), _hit(0.80, "book-a")]
    without_override = select_adaptive_pdf_attachment_count(
        question="simple question",
        hits=hits,
        effective_max=3,
        max_attachments=3,
        min_attachments=1,
        adaptive_enabled=True,
        adaptive_applicable=True,
        top_score_low=0.42,
        top_score_very_low=0.30,
        score_gap_low=0.003,
        complexity_length_tokens=25,
    )
    assert without_override.count == 2

    with_override = select_adaptive_pdf_attachment_count(
        question="simple question",
        hits=hits,
        effective_max=3,
        max_attachments=3,
        min_attachments=1,
        adaptive_enabled=True,
        adaptive_applicable=True,
        top_score_low=0.42,
        top_score_very_low=0.30,
        score_gap_low=0.003,
        complexity_length_tokens=25,
        score_gap_override=0.01,
    )
    assert with_override.count == 1


def test_selector_score_gap_override_boundary_uses_new_threshold() -> None:
    hits = [_hit(0.90, "book-a"), _hit(0.899, "book-a"), _hit(0.80, "book-a")]
    equal_threshold = select_adaptive_pdf_attachment_count(
        question="simple question",
        hits=hits,
        effective_max=3,
        max_attachments=3,
        min_attachments=1,
        adaptive_enabled=True,
        adaptive_applicable=True,
        top_score_low=0.42,
        top_score_very_low=0.30,
        score_gap_low=0.003,
        complexity_length_tokens=25,
        score_gap_override=0.003,
    )
    assert equal_threshold.count == 1

    just_below_threshold = select_adaptive_pdf_attachment_count(
        question="simple question",
        hits=hits,
        effective_max=3,
        max_attachments=3,
        min_attachments=1,
        adaptive_enabled=True,
        adaptive_applicable=True,
        top_score_low=0.42,
        top_score_very_low=0.30,
        score_gap_low=0.003,
        complexity_length_tokens=25,
        score_gap_override=0.0029,
    )
    assert just_below_threshold.count == 2


def test_adaptive_attachable_hits_zero_returns_zero() -> None:
    hits = [_hit(0.95, "book-a"), _hit(0.90, "book-a")]
    decision = _decide(question="simple question", hits=hits, effective_max=0)
    assert decision.count == 0


def test_adaptive_empty_hits_returns_zero() -> None:
    decision = _decide(question="simple question", hits=[], effective_max=0)
    assert decision.count == 0


def test_adaptive_min_attachments_above_effective_max_caps_to_effective() -> None:
    hits = [_hit(0.95, "book-a"), _hit(0.70, "book-a")]
    decision = _decide(
        question="simple question",
        hits=hits,
        effective_max=1,
        min_attachments=2,
        max_attachments=3,
    )
    assert decision.count == 1


def test_adaptive_disabled_matches_fixed_behavior() -> None:
    hits = [_hit(0.95, "book-a"), _hit(0.90, "book-a"), _hit(0.80, "book-a")]
    decision = _decide(
        question="simple question",
        hits=hits,
        effective_max=2,
        adaptive_enabled=False,
        max_attachments=3,
    )
    assert decision.count == 2
    assert "adaptive_disabled" in decision.reason_flags


def test_non_target_intent_matches_fixed_behavior() -> None:
    hits = [_hit(0.95, "book-a"), _hit(0.90, "book-a"), _hit(0.80, "book-a")]
    decision = _decide(
        question="simple question",
        hits=hits,
        effective_max=2,
        adaptive_enabled=True,
        adaptive_applicable=False,
        max_attachments=3,
    )
    assert decision.count == 2
    assert "adaptive_disabled" in decision.reason_flags


def test_pdf_window_plugin_logs_warning_when_score_scale_is_not_normalized(tmp_path: Path, caplog) -> None:
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    plugin = PdfWindowGroundingContextPlugin(
        max_attachments=3,
        adaptive_enabled=True,
    )
    hit = Hit(
        document=Document(
            id="doc-1",
            text="evidence",
            metadata={
                "source_id": "book-a",
                "source_pdf_path": str(pdf_path),
                "start_page": 1,
                "end_page": 1,
                "page": 1,
            },
        ),
        score=1.2,
    )

    caplog.set_level("WARNING")
    attached_hits, attachments = plugin._attachments_for_hits(
        question="explain topic",
        intent="grounded_textbook",
        hits=[hit],
    )
    assert len(attached_hits) == 1
    assert len(attachments) == 1
    assert "normalized retrieval scores" in caplog.text


def test_pdf_window_plugin_uses_pre_expansion_gap_signal_for_decision(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    plugin = PdfWindowGroundingContextPlugin(
        max_attachments=3,
        adaptive_enabled=True,
        adaptive_score_gap_low=0.003,
    )
    hits = [
        Hit(
            document=Document(
                id="doc-1",
                text="evidence 1",
                metadata={
                    "source_id": "book-a",
                    "source_pdf_path": str(pdf_path),
                    "start_page": 1,
                    "end_page": 1,
                    "page": 1,
                },
            ),
            score=0.90,
        ),
        Hit(
            document=Document(
                id="doc-2",
                text="evidence 2",
                metadata={
                    "source_id": "book-a",
                    "source_pdf_path": str(pdf_path),
                    "start_page": 2,
                    "end_page": 2,
                    "page": 2,
                },
            ),
            score=0.899,
        ),
    ]

    _, attachments_without_signal = plugin._attachments_for_hits(
        question="simple question",
        intent="grounded_textbook",
        hits=hits,
    )
    assert len(attachments_without_signal) == 2

    _, attachments_with_signal = plugin._attachments_for_hits(
        question="simple question",
        intent="grounded_textbook",
        hits=hits,
        retrieval_signals=AdaptiveAttachmentRetrievalSignals(
            top_score_pre_expansion=0.90,
            score_gap_pre_expansion=0.01,
            top_score_post_expansion=0.90,
            score_gap_post_expansion=0.001,
        ),
    )
    assert len(attachments_with_signal) == 1
