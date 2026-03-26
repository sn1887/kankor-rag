from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.openai_compat import router as openai_compat_router
from .routes.whatsapp import router as whatsapp_router
from .settings import load_settings
from .wiring import get_app_state

settings = load_settings()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    state = get_app_state()
    if state.whatsapp is not None:
        await state.whatsapp.start()
    try:
        yield
    finally:
        if state.whatsapp is not None:
            await state.whatsapp.stop()


app = FastAPI(
    title='Kankor RAG API',
    version='0.1.0',
    summary='Thin HTTP layer over a modular RAG engine.',
    lifespan=_lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(openai_compat_router)
app.include_router(whatsapp_router)

@app.get('/')
def root() -> dict:
    state = get_app_state()
    return {'name': 'kankor-rag-api', 'status': 'ok', 'documents_loaded': state.pipeline.vector_store.size}
