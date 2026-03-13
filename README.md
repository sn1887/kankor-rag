# Kankor RAG Space

A modular Retrieval-Augmented Generation (RAG) system for Afghanistan's Kankor exam content.
The repository is structured for local development and deployment as a Hugging Face Docker Space.

## Architecture

- OpenWebUI as the default interface (connected through OpenAI-compatible API routes)
- FastAPI backend (`apps/api`) with both custom SSE (`/v1/chat/stream`) and OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) endpoints
- Optional Next.js frontend (`apps/web`) for custom source-panel UX
- Reusable core package (`packages/rag_core`) with pluggable embeddings + LLM providers (`hash`, `e5`, `openai`, `gemini`, `deepseek`, `transformers`)
- Offline data pipeline (`scripts/`) for PDF ingestion, corpus audit, and FAISS index build
- Docker startup (`docker/`) with OpenWebUI-first API mode and optional Next.js mode

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ./packages/rag_core[serve] -e ./apps/api

cd apps/web
npm install
cd ../..
```

## OpenWebUI (Default Interface)

Run OpenWebUI + API together (recommended):

```bash
cd docker
cp .env.example .env
RAG_OPENAI_COMPAT_API_KEY=changeme docker compose -f docker-compose.openwebui.yml up --build
```

Then open `http://127.0.0.1:3000`.

The API is exposed at `http://127.0.0.1:8000/v1`, and OpenWebUI uses the OpenAI-compatible endpoints automatically.

## Quick Demo (Local)

Build a small demo index with hashing embeddings:

```bash
python scripts/build_index.py \
  --input data/sample_corpus/kankor_sample.jsonl \
  --output-dir data/sample_index \
  --embedding-backend hash
```

Start the API:

```bash
export RAG_INDEX_PATH=data/sample_index/index.faiss
export RAG_DOCSTORE_PATH=data/sample_index/metadata.jsonl
export RAG_EMBEDDING_BACKEND=hash
export RAG_DEMO_MODE=true
uvicorn kankor_api.main:app --host 127.0.0.1 --port 8000
```

Start the web app in a second terminal:

```bash
cd apps/web
BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

The Next.js app is now optional. Use it when you want the custom source panel; otherwise use OpenWebUI as the default interface.

## OpenAI Setup (LLM + Embeddings)

Build index with OpenAI embeddings and keep vectors in local FAISS files:

```bash
python scripts/build_index.py \
  --input data/corpus/kankor_corpus.jsonl \
  --output-dir data/index/kankor_openai \
  --embedding-backend openai \
  --openai-embedding-model-id text-embedding-3-small \
  --corpus-version kankor-corpus@2026.03
```

Run API with OpenAI generation + OpenAI query embeddings:

```bash
export RAG_INDEX_PATH=data/index/kankor_openai/index.faiss
export RAG_DOCSTORE_PATH=data/index/kankor_openai/metadata.jsonl
export RAG_LLM_BACKEND=openai
export RAG_EMBEDDING_BACKEND=openai
export RAG_OPENAI_MODEL_ID=gpt-4o-mini
export RAG_OPENAI_EMBEDDING_MODEL_ID=text-embedding-3-small
export OPENAI_API_KEY=<your-key>
uvicorn kankor_api.main:app --host 127.0.0.1 --port 8000
```

## Gemini Setup (LLM + Embeddings)

Build index with Gemini embeddings:

```bash
python scripts/build_index.py \
  --input data/corpus/kankor_corpus.jsonl \
  --output-dir data/index/kankor_gemini \
  --embedding-backend gemini \
  --gemini-embedding-model-id text-embedding-004 \
  --corpus-version kankor-corpus@2026.03
```

Run API with Gemini generation + Gemini query embeddings:

```bash
export RAG_INDEX_PATH=data/index/kankor_gemini/index.faiss
export RAG_DOCSTORE_PATH=data/index/kankor_gemini/metadata.jsonl
export RAG_LLM_BACKEND=gemini
export RAG_EMBEDDING_BACKEND=gemini
export RAG_GEMINI_MODEL_ID=gemini-2.0-flash
export RAG_GEMINI_EMBEDDING_MODEL_ID=text-embedding-004
export GEMINI_API_KEY=<your-key>
uvicorn kankor_api.main:app --host 127.0.0.1 --port 8000
```

## DeepSeek Setup (LLM + Embeddings)

Build index with DeepSeek embeddings:

```bash
python scripts/build_index.py \
  --input data/corpus/kankor_corpus.jsonl \
  --output-dir data/index/kankor_deepseek \
  --embedding-backend deepseek \
  --deepseek-embedding-model-id deepseek-embedding \
  --corpus-version kankor-corpus@2026.03
```

Run API with DeepSeek generation + DeepSeek query embeddings:

```bash
export RAG_INDEX_PATH=data/index/kankor_deepseek/index.faiss
export RAG_DOCSTORE_PATH=data/index/kankor_deepseek/metadata.jsonl
export RAG_LLM_BACKEND=deepseek
export RAG_EMBEDDING_BACKEND=deepseek
export RAG_DEEPSEEK_MODEL_ID=deepseek-chat
export RAG_DEEPSEEK_EMBEDDING_MODEL_ID=deepseek-embedding
export DEEPSEEK_API_KEY=<your-key>
uvicorn kankor_api.main:app --host 127.0.0.1 --port 8000
```

## Corpus Pipeline

### 1) Extract PDFs to JSONL

Extract textbook content from `data/raw_pdfs`:

```bash
python scripts/pdf_to_jsonl.py \
  --input-dir data/raw_pdfs \
  --output data/corpus/kankor_corpus.jsonl \
  --corpus-version kankor-corpus-2026.03 \
  --ocr-fallback
```

Useful outputs:
- `data/corpus/kankor_corpus.jsonl`
- `data/corpus/unreadable_pages.jsonl`
- `data/corpus/page_extraction_audit.jsonl`

### 2) Audit corpus quality

```bash
python scripts/corpus_audit.py \
  --input data/corpus/kankor_corpus.jsonl \
  --chunk-size 140 \
  --chunk-overlap 24 \
  --show-samples 8
```

### 3) Build FAISS index

```bash
python scripts/build_index.py \
  --input data/corpus/kankor_corpus.jsonl \
  --output-dir data/index/kankor_e5_base \
  --embedding-backend e5 \
  --embedding-model-id intfloat/multilingual-e5-base \
  --embedding-batch-size 16 \
  --corpus-version kankor-corpus@2026.03
```

## Runtime Configuration

Key environment variables:

- `RAG_INDEX_PATH` and `RAG_DOCSTORE_PATH`: FAISS and metadata paths
- `RAG_VECTOR_STORE_BACKEND`: vector backend key (`faiss`) or `module.path:factory`
- `RAG_LLM_BACKEND`: `transformers`, `openai`, `gemini`, `deepseek`, or `module.path:factory`
- `RAG_EMBEDDING_BACKEND`: `hash`, `e5`, `openai`, `gemini`, `deepseek`, or `module.path:factory`
- `RAG_MODEL_ID`: Hugging Face model id for local transformers generation
- `RAG_EMBEDDING_MODEL_ID`: local E5 embedding model id
- `RAG_OPENAI_MODEL_ID`: OpenAI model id for chat generation
- `RAG_OPENAI_EMBEDDING_MODEL_ID`: OpenAI embeddings model id
- `RAG_OPENAI_API_KEY`: optional key override (`OPENAI_API_KEY` also works)
- `RAG_OPENAI_BASE_URL`: optional OpenAI-compatible endpoint
- `RAG_OPENAI_TIMEOUT_SECONDS`: timeout for OpenAI requests
- `RAG_OPENAI_EMBEDDING_DIMENSIONS`: optional output dimensions for OpenAI embeddings
- `RAG_GEMINI_MODEL_ID` / `RAG_GEMINI_EMBEDDING_MODEL_ID`: Gemini model ids
- `RAG_GEMINI_API_KEY`: Gemini key override (falls back to `GEMINI_API_KEY`)
- `RAG_GEMINI_BASE_URL`: Gemini OpenAI-compatible base URL (default `https://generativelanguage.googleapis.com/v1beta/openai`)
- `RAG_GEMINI_TIMEOUT_SECONDS`: timeout for Gemini requests
- `RAG_GEMINI_EMBEDDING_DIMENSIONS`: optional output dimensions for Gemini embeddings
- `RAG_DEEPSEEK_MODEL_ID` / `RAG_DEEPSEEK_EMBEDDING_MODEL_ID`: DeepSeek model ids
- `RAG_DEEPSEEK_API_KEY`: DeepSeek key override (falls back to `DEEPSEEK_API_KEY`)
- `RAG_DEEPSEEK_BASE_URL`: DeepSeek OpenAI-compatible base URL (default `https://api.deepseek.com/v1`)
- `RAG_DEEPSEEK_TIMEOUT_SECONDS`: timeout for DeepSeek requests
- `RAG_DEEPSEEK_EMBEDDING_DIMENSIONS`: optional output dimensions for DeepSeek embeddings
- `RAG_OPENAI_COMPAT_API_KEY`: optional Bearer token for `/v1/models` and `/v1/chat/completions`
- `RAG_CHAT_API_KEY`: optional Bearer token for `/v1/chat/stream` (falls back to `RAG_OPENAI_COMPAT_API_KEY`)
- `RAG_SOURCE_PDF_URL_TEMPLATE`: citation URL template for exact textbook pages (`{grade_band}`, `{source_id}`, `{page}`)
- `RAG_ALLOW_HASH_EMBEDDER_FALLBACK`: `false` by default; set `true` only for explicit degraded-mode tolerance
- `RAG_MAX_NEW_TOKENS_HARD_LIMIT`: hard upper bound enforced on per-request `max_tokens`
- `RAG_TEMPERATURE_MIN` / `RAG_TEMPERATURE_MAX`: allowed request temperature range
- `RAG_UI_INTERFACE`: `openwebui` (default, API-only container) or `nextjs` (runs API + Next.js in one container)
- `RAG_DEMO_MODE`: set `true` to bypass local model generation
- `RAG_GENERATION_MODE`: `sample`, `greedy`, or `contrastive`
- `RAG_CONTRASTIVE_PENALTY_ALPHA`: contrastive decoding parameter
- `RAG_CONTRASTIVE_TOP_K`: contrastive decoding parameter
- `BACKEND_API_KEY` (Next.js only): forwards `Authorization: Bearer ...` to backend `/v1/chat/stream`

Contrastive decoding requires remote generation code from:
`transformers-community/contrastive-search`

## Hugging Face Space Deployment

The Docker entrypoint supports artifact download at startup:

- `RAG_DOWNLOAD_ON_BOOT=true`
- `RAG_DATASET_REPO_ID=<hf-dataset-repo>`
- `RAG_DATASET_SUBFOLDER=<optional-subfolder>`
- `HF_TOKEN=<optional, for private dataset>`

When enabled, the container downloads index artifacts and sets:
- `RAG_INDEX_PATH`
- `RAG_DOCSTORE_PATH`

Recommended low-cost path:

1. Verify locally with local FAISS index files.
2. Deploy public demo to Hugging Face Spaces CPU Basic.
3. Keep vectors in FAISS artifacts until update frequency or filtering needs justify moving to pgvector.

## Docs

- `docs/REPOSITORY.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA.md`
- `docs/SECURITY.md`
