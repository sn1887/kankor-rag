from fastapi import APIRouter
from ..wiring import get_app_state
router = APIRouter(prefix='/v1/health', tags=['health'])

@router.get('')
def health() -> dict:
    state = get_app_state()
    return {'status': 'ok', 'model_id': state.settings.rag_model_id, 'embedding_backend': state.settings.rag_embedding_backend, 'corpus_version': state.settings.rag_corpus_version, 'documents_loaded': state.pipeline.vector_store.size}
