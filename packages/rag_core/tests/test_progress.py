from __future__ import annotations

from rag_core.contracts.progress import NoopProgressSink, ProgressEvent, ProgressStage
from rag_core.rag.progress import LocalizedProgressPresenter, PipelineProgressReporter, resolve_progress_locale


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.closed = False

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def test_resolve_progress_locale_matches_detected_pashto() -> None:
    locale = resolve_progress_locale(question="په کوم فصل کې د نیوتن قانونونه دي؟", default_language="match-user")
    assert locale == "ps"


def test_localized_progress_presenter_routes_english_query_to_dari_labels() -> None:
    presenter = LocalizedProgressPresenter(question="Which chapter covers photosynthesis?", default_language="match-user")
    assert presenter.message_for(ProgressStage.RETRIEVING) == "در حال بازیابی منبع‌ها..."
    assert presenter.message_for(ProgressStage.DONE) == "تمام شد."


def test_noop_progress_sink_is_safe() -> None:
    sink = NoopProgressSink()
    sink.emit(ProgressEvent(stage=ProgressStage.THINKING, message="Thinking..."))
    sink.close()


def test_pipeline_progress_reporter_emits_unique_stage_sequence_and_closes_sink() -> None:
    sink = _RecordingSink()
    reporter = PipelineProgressReporter(
        sink=sink,
        question="فصل یازدهم درباره چیست؟",
        default_language="match-user",
    )

    reporter.thinking()
    reporter.thinking()
    reporter.retrieving()
    reporter.reading()
    reporter.writing()
    reporter.done()
    reporter.error(metadata={"message": "ignored"})
    reporter.close()

    assert [event.stage for event in sink.events] == [
        ProgressStage.THINKING,
        ProgressStage.RETRIEVING,
        ProgressStage.READING,
        ProgressStage.WRITING,
        ProgressStage.DONE,
    ]
    assert sink.events[-1].done is True
    assert sink.closed is True
