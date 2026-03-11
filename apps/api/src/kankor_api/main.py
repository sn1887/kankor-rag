from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .settings import Settings
from .wiring import get_app_state

settings = Settings()
app = FastAPI(title='Kankor RAG API', version='0.1.0', summary='Thin HTTP layer over a modular RAG engine.')
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
app.include_router(health_router)
app.include_router(chat_router)

@app.get('/')
def root() -> dict:
    state = get_app_state()
    return {'name': 'kankor-rag-api', 'status': 'ok', 'documents_loaded': state.pipeline.vector_store.size}
