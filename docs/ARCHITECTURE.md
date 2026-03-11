# Architecture

## Request flow

1. User types a question in the Next.js UI.
2. Browser posts to `/api/chat`.
3. Next.js proxies to the internal FastAPI service.
4. FastAPI extracts the latest user question and passes it to `RAGPipeline`.
5. The pipeline embeds the query, runs FAISS similarity search, packages source metadata, builds a grounded prompt, and streams generated text tokens.
6. FastAPI emits SSE events: `sources`, `delta`, `done`.
7. The web app renders the answer and source panel incrementally.

## Why this shape works on free tier

- Expensive indexing happens offline.
- Online serving uses a prebuilt local FAISS index.
- The public web process and private API process share a single container.
- The Space repo stays small because large corpus/index artifacts can live elsewhere.

## Scaling path without rewrites

- replace the LLM provider with hosted inference
- shard artifacts by subject or corpus version
- replace FAISS with a managed vector database
- keep the API schema and UI intact
