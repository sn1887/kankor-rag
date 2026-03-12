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

A modular Retrieval-Augmented Generation (RAG) repository for Afghanistan's Kankor exam, designed from the provided blueprint for a Hugging Face Docker Space on the free CPU tier.

## Stack

- Next.js chat UI with streaming responses and a sources panel
- FastAPI backend that performs retrieval and streams answer tokens
- Reusable `rag_core` package with swappable interfaces for LLMs, embeddings, vector stores, and corpus sources
- Offline tooling to build FAISS indexes from a versioned corpus
- Sample Kankor corpus for local smoke tests
- Docker wiring for a single-container Hugging Face Space

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ./packages/rag_core[serve] -e ./apps/api

cd apps/web
npm install
cd ../..
```

Build a tiny demo index without downloading a multilingual embedding model:

```bash
python scripts/build_index.py   --input data/sample_corpus/kankor_sample.jsonl   --output-dir data/sample_index   --embedding-backend hash
```

Run the backend:

```bash
export RAG_INDEX_PATH=data/sample_index/index.faiss
export RAG_DOCSTORE_PATH=data/sample_index/metadata.json
export RAG_EMBEDDING_BACKEND=hash
export RAG_DEMO_MODE=true
uvicorn kankor_api.main:app --host 127.0.0.1 --port 8000
```

Run the web app:

```bash
cd apps/web
BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

## High-fidelity PDF extraction

For difficult textbooks with broken text layers, multi-column layouts, tables, or equations, the repo now supports an optional `marker` extractor in `scripts/pdf_to_jsonl.py`. Marker is the preferred upgrade over wiring Surya directly, since it already uses layout-aware OCR and renders markdown with tables and math preserved.

Install Marker:

```bash
python -m pip install marker-pdf
```

Run the extractor with Marker:

```bash
python scripts/pdf_to_jsonl.py \
  --input-dir data/raw_pdfs \
  --output data/corpus/kankor_corpus.jsonl \
  --extractor marker \
  --marker-force-ocr
```

If you want to stay with the lighter built-in pipeline, `--extractor hybrid` remains the default and now supports stronger OCR controls such as `--ocr-psm`, `--ocr-oem`, `--extract-tables`, and `--bidi-native`.

## Production-shaped defaults

- LLM: `Qwen/Qwen3.5-2B`
- Embeddings: `intfloat/multilingual-e5-small`
- Vector store: FAISS loaded from prebuilt artifacts
- Corpus policy: versioned, auditable, multilingual, source-first

## Docs

- `docs/REPOSITORY.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA.md`
- `docs/SECURITY.md`
