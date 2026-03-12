from fastapi import APIRouter
from ..wiring import get_app_state
router = APIRouter(prefix='/v1/health', tags=['health'])

@router.get('')
def health() -> dict:
    state = get_app_state()
    llm_model = state.settings.rag_openai_model_id if state.settings.rag_llm_backend.lower() == 'openai' else state.settings.rag_model_id
    embedding_model = state.settings.rag_openai_embedding_model_id if state.settings.rag_embedding_backend.lower() == 'openai' else state.settings.rag_embedding_model_id
    return {
        'status': 'ok',
        'llm_backend': state.settings.rag_llm_backend,
        'llm_model_id': llm_model,
        'embedding_backend': state.settings.rag_embedding_backend,
        'embedding_model_id': embedding_model,
        'corpus_version': state.settings.rag_corpus_version,
        'documents_loaded': state.pipeline.vector_store.size,
    }
