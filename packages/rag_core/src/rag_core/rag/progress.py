from __future__ import annotations

from rag_core.contracts.progress import NoopProgressSink, ProgressEvent, ProgressSink, ProgressStage
from rag_core.rag.prompts import detect_answer_language


_DISPLAY_LANGUAGE_TO_LOCALE = {
    "دری": "fa",
    "پښتو": "ps",
    "english": "fa",
    "العربية": "fa",
}

_LOCALE_ALIASES = {
    "fa": "fa",
    "fa-af": "fa",
    "dari": "fa",
    "persian": "fa",
    "ps": "ps",
    "pashto": "ps",
    "ar": "fa",
    "arabic": "fa",
    "en": "fa",
    "english": "fa",
}

_STATUS_MESSAGES: dict[str, dict[ProgressStage, str]] = {
    "en": {
        ProgressStage.THINKING: "Thinking...",
        ProgressStage.RETRIEVING: "Retrieving docs...",
        ProgressStage.READING: "Reading sources...",
        ProgressStage.WRITING: "Writing answer...",
        ProgressStage.DONE: "Done.",
        ProgressStage.ERROR: "Something went wrong.",
    },
    "fa": {
        ProgressStage.THINKING: "در حال فکر کردن...",
        ProgressStage.RETRIEVING: "در حال بازیابی منبع‌ها...",
        ProgressStage.READING: "در حال خواندن منبع‌ها...",
        ProgressStage.WRITING: "در حال نوشتن پاسخ...",
        ProgressStage.DONE: "تمام شد.",
        ProgressStage.ERROR: "مشکلی پیش آمد.",
    },
    "ps": {
        ProgressStage.THINKING: "اوس فکر کوم...",
        ProgressStage.RETRIEVING: "اړوند متنونه رااخلم...",
        ProgressStage.READING: "سرچينې لولم...",
        ProgressStage.WRITING: "ځواب ليکم...",
        ProgressStage.DONE: "بشپړ شو.",
        ProgressStage.ERROR: "یوه ستونزه پېښه شوه.",
    },
    "ar": {
        ProgressStage.THINKING: "أفكر الآن...",
        ProgressStage.RETRIEVING: "أسترجع المصادر...",
        ProgressStage.READING: "أقرأ المصادر...",
        ProgressStage.WRITING: "أكتب الإجابة...",
        ProgressStage.DONE: "تم.",
        ProgressStage.ERROR: "حدث خطأ ما.",
    },
}


def resolve_progress_locale(*, question: str, default_language: str = "auto") -> str:
    normalized = (default_language or "").strip().lower()
    if normalized in _LOCALE_ALIASES:
        return _LOCALE_ALIASES[normalized]
    display_language = detect_answer_language(question, default_language).strip().lower()
    return _DISPLAY_LANGUAGE_TO_LOCALE.get(display_language, "fa")


class LocalizedProgressPresenter:
    def __init__(self, *, question: str, default_language: str = "auto") -> None:
        self.locale = resolve_progress_locale(question=question, default_language=default_language)

    def message_for(self, stage: ProgressStage) -> str:
        return _STATUS_MESSAGES.get(self.locale, _STATUS_MESSAGES["fa"]).get(stage, _STATUS_MESSAGES["fa"][stage])

    def event_for(
        self,
        stage: ProgressStage,
        *,
        done: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> ProgressEvent:
        return ProgressEvent(
            stage=stage,
            message=self.message_for(stage),
            done=done,
            metadata=dict(metadata or {}),
        )


class PipelineProgressReporter:
    def __init__(
        self,
        *,
        sink: ProgressSink | None,
        question: str,
        default_language: str = "auto",
    ) -> None:
        self._sink: ProgressSink = sink or NoopProgressSink()
        self._presenter = LocalizedProgressPresenter(
            question=question,
            default_language=default_language,
        )
        self._current_stage: ProgressStage | None = None
        self._finalized = False

    def emit(
        self,
        stage: ProgressStage,
        *,
        done: bool = False,
        metadata: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        if self._finalized and not force:
            return
        if not force and stage == self._current_stage:
            return
        self._sink.emit(self._presenter.event_for(stage, done=done, metadata=metadata))
        self._current_stage = stage
        if done or stage in {ProgressStage.DONE, ProgressStage.ERROR}:
            self._finalized = True

    def thinking(self, *, metadata: dict[str, object] | None = None) -> None:
        self.emit(ProgressStage.THINKING, metadata=metadata)

    def retrieving(self, *, metadata: dict[str, object] | None = None) -> None:
        self.emit(ProgressStage.RETRIEVING, metadata=metadata)

    def reading(self, *, metadata: dict[str, object] | None = None) -> None:
        self.emit(ProgressStage.READING, metadata=metadata)

    def writing(self, *, metadata: dict[str, object] | None = None) -> None:
        self.emit(ProgressStage.WRITING, metadata=metadata)

    def done(self, *, metadata: dict[str, object] | None = None) -> None:
        self.emit(ProgressStage.DONE, done=True, metadata=metadata, force=self._current_stage != ProgressStage.DONE)

    def error(self, *, metadata: dict[str, object] | None = None) -> None:
        if self._finalized:
            return
        self.emit(ProgressStage.ERROR, done=True, metadata=metadata, force=True)

    def close(self) -> None:
        self._sink.close()
