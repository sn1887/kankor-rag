# Architecture

## Request flow

1. User chats through OpenWebUI (default) or the optional Next.js UI.
2. OpenWebUI calls FastAPI OpenAI-compatible endpoints (`/v1/models`, `/v1/chat/completions`) or Next.js proxies to `/v1/chat/stream`.
3. FastAPI extracts the latest user question and passes it to `RAGPipeline`.
4. The pipeline embeds the query, runs FAISS similarity search, packages source metadata, builds a grounded prompt, and streams generated text tokens.
5. Embedding and LLM adapters are selected by env-driven backend settings (`hash` / `e5` / `openai` / `gemini` / `deepseek` and `transformers` / `openai` / `gemini` / `deepseek`).
6. FastAPI streams OpenAI-style chunks for OpenWebUI or custom SSE events for the Next.js app.
7. The selected UI renders the answer incrementally.

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
