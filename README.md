---
title: Kankor RAG Space
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Kankor RAG Space

A modular Retrieval-Augmented Generation (RAG) system for Afghanistan's Kankor exam content.
The repository is structured for local development and deployment as a Hugging Face Docker Space.

## Architecture

- Next.js frontend (`apps/web`) with streaming chat UI and source panel
- FastAPI backend (`apps/api`) with SSE streaming endpoint
- Reusable core package (`packages/rag_core`) for embeddings, vector store, and LLM providers
- Offline data pipeline (`scripts/`) for PDF ingestion, corpus audit, and FAISS index build
- Docker startup (`docker/`) for running web + API in one container

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
export RAG_DOCSTORE_PATH=data/sample_index/metadata.json
export RAG_EMBEDDING_BACKEND=hash
export RAG_DEMO_MODE=true
uvicorn kankor_api.main:app --host 127.0.0.1 --port 8000
```

Start the web app in a second terminal:

```bash
cd apps/web
BACKEND_URL=http://127.0.0.1:8000 npm run dev
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
- `RAG_MODEL_ID`: Hugging Face model id for local generation
- `RAG_EMBEDDING_MODEL_ID`: embedding model id
- `RAG_DEMO_MODE`: set `true` to bypass local model generation
- `RAG_GENERATION_MODE`: `sample`, `greedy`, or `contrastive`
- `RAG_CONTRASTIVE_PENALTY_ALPHA`: contrastive decoding parameter
- `RAG_CONTRASTIVE_TOP_K`: contrastive decoding parameter

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

## Docs

- `docs/REPOSITORY.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA.md`
- `docs/SECURITY.md`
