from __future__ import annotations

from rag_core.rag.intent_router import IntentRouter, RAGIntent


def test_intent_router_classifies_smalltalk() -> None:
    router = IntentRouter()
    route = router.route(question="سلام")
    assert route.intent == RAGIntent.SMALLTALK
    assert route.retrieval_queries == []


def test_intent_router_classifies_study_coach() -> None:
    router = IntentRouter()
    route = router.route(question="How should I build a daily study schedule for Kankor?")
    assert route.intent == RAGIntent.STUDY_COACH
    assert route.retrieval_queries == []


def test_intent_router_classifies_practice_generation() -> None:
    router = IntentRouter()
    route = router.route(question="از فصل حرکت 5 سوال چهارگزینه‌ای بساز")
    assert route.intent == RAGIntent.PRACTICE_GENERATION
    assert route.retrieval_queries


def test_intent_router_classifies_topic_locator_with_decomposition() -> None:
    router = IntentRouter(max_decomposition_queries=4)
    route = router.route(question="Where is motion taught in the grade 10 textbook?")
    assert route.intent == RAGIntent.TOPIC_LOCATOR
    assert route.is_broad
    assert len(route.retrieval_queries) >= 2


def test_intent_router_defaults_to_grounded_textbook() -> None:
    router = IntentRouter()
    route = router.route(question="Explain kinetic energy with one simple example.")
    assert route.intent == RAGIntent.GROUNDED_TEXTBOOK
    assert route.retrieval_queries


def test_intent_router_classifies_chapter_title_as_topic_locator() -> None:
    router = IntentRouter()
    route = router.route(question="فصل دوم کتاب فزیک چی است؟")
    assert route.intent == RAGIntent.TOPIC_LOCATOR
    assert any(("فصل" in query or "chapter" in query) and "2" in query for query in route.retrieval_queries)


def test_intent_router_question_sheet_avoids_full_ocr_blob_query() -> None:
    router = IntentRouter(max_decomposition_queries=5)
    question = (
        "Solve the questions from image.\n\n"
        "Extracted text:\n"
        "1) Explain Newton's second law and solve an example.\n"
        "2) Calculate velocity for distance/time.\n"
        "3) Write the formula for acceleration."
    )
    route = router.route(question=question)
    assert route.intent == RAGIntent.DIRECT_SOLVER
    assert route.retrieval_queries == []


def test_intent_router_classifies_self_contained_stem_problem_as_direct_solver() -> None:
    router = IntentRouter()
    route = router.route(question="Solve for x: 2x + 5 = 17")
    assert route.intent == RAGIntent.DIRECT_SOLVER
    assert route.retrieval_queries == []


def test_intent_router_keeps_non_self_contained_stem_query_grounded() -> None:
    router = IntentRouter()
    route = router.route(question="Explain Newton's second law from grade 10 physics textbook.")
    assert route.intent == RAGIntent.GROUNDED_TEXTBOOK
    assert route.retrieval_queries
