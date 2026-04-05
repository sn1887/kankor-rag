from __future__ import annotations

from rag_core.rag.intent_router import IntentRouter, RAGIntent


def test_intent_router_classifies_smalltalk() -> None:
    router = IntentRouter()
    route = router.route(question="سلام")
    assert route.intent == RAGIntent.SMALLTALK
    assert route.retrieval_queries == []


def test_intent_router_classifies_practice_generation() -> None:
    router = IntentRouter()
    route = router.route(question="از فصل حرکت 5 سوال چهارگزینه‌ای بساز")
    assert route.intent == RAGIntent.PRACTICE_GENERATION
    assert route.retrieval_queries


def test_intent_router_defaults_to_grounded_textbook() -> None:
    router = IntentRouter()
    route = router.route(question="Explain kinetic energy with one simple example.")
    assert route.intent == RAGIntent.GROUNDED_TEXTBOOK
    assert route.retrieval_queries


def test_intent_router_location_style_queries_stay_grounded_for_structure_lookup_helper() -> None:
    router = IntentRouter(max_decomposition_queries=4)
    route = router.route(question="Where is motion taught in the grade 10 textbook?")
    assert route.intent == RAGIntent.GROUNDED_TEXTBOOK
    assert route.retrieval_queries


def test_intent_router_solver_style_queries_stay_grounded() -> None:
    router = IntentRouter()
    route = router.route(question="Solve for x: 2x + 5 = 17")
    assert route.intent == RAGIntent.GROUNDED_TEXTBOOK
    assert route.retrieval_queries


def test_intent_router_study_schedule_queries_stay_grounded() -> None:
    router = IntentRouter()
    route = router.route(question="How should I build a daily study schedule for Kankor?")
    assert route.intent == RAGIntent.GROUNDED_TEXTBOOK
    assert route.retrieval_queries
