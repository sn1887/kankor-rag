from fastapi import APIRouter
from ..wiring import get_app_state
router = APIRouter(prefix='/v1/health', tags=['health'])

@router.get('')
def health() -> dict:
    state = get_app_state()
    llm_model = state.settings.rag_openai_model_id if state.settings.rag_llm_backend.lower() == 'openai' else state.settings.rag_model_id
    configured_backend = state.settings.rag_embedding_backend.lower().strip()
    embedder = state.pipeline.embedder
    runtime_embedding_backend = configured_backend
    runtime_backend_getter = getattr(embedder, 'runtime_backend', None)
    if callable(runtime_backend_getter):
        runtime_embedding_backend = str(runtime_backend_getter())
    fallback_reason = getattr(embedder, 'fallback_reason', None)

    if runtime_embedding_backend == 'hash' and configured_backend == 'e5':
        embedding_model = 'hash-fallback'
    elif configured_backend == 'openai':
        embedding_model = state.settings.rag_openai_embedding_model_id
    elif configured_backend == 'hash':
        embedding_model = 'hash'
    else:
        embedding_model = state.settings.rag_embedding_model_id

    return {
        'status': 'ok',
        'llm_backend': state.settings.rag_llm_backend,
        'llm_model_id': llm_model,
        'embedding_backend': state.settings.rag_embedding_backend,
        'embedding_runtime_backend': runtime_embedding_backend,
        'embedding_model_id': embedding_model,
        'embedding_fallback_reason': fallback_reason,
        'vector_store_backend': state.settings.rag_vector_store_backend,
        'corpus_version': state.settings.rag_corpus_version,
        'documents_loaded': state.pipeline.vector_store.size,
    }
