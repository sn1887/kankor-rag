from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rag_core.contracts.embeddings import Embedder
from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource
from rag_core.impl.embeddings_deepseek import DeepSeekEmbedder
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.embeddings_gemini import GeminiEmbedder
from rag_core.impl.embeddings_openai import OpenAIEmbedder
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.util.text_splitter import chunk_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a FAISS index from a JSONL corpus or HF dataset.')
    parser.add_argument('--input', help='Path to a local JSONL corpus file.', default=None)
    parser.add_argument('--dataset-name', help='Optional HF dataset repo id.', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--embedding-backend', choices=['e5', 'hash', 'openai', 'gemini', 'deepseek'], default='e5')
    parser.add_argument('--embedding-model-id', default='intfloat/multilingual-e5-small')
    parser.add_argument('--embedding-batch-size', type=int, default=128)

    parser.add_argument('--openai-embedding-model-id', default='text-embedding-3-small')
    parser.add_argument('--openai-api-key', default=None, help='Optional OpenAI API key override. Defaults to OPENAI_API_KEY env var.')
    parser.add_argument('--openai-base-url', default=None, help='Optional OpenAI-compatible base URL.')
    parser.add_argument('--openai-timeout-seconds', type=float, default=120.0)
    parser.add_argument('--openai-embedding-dimensions', type=int, default=None)

    parser.add_argument('--gemini-embedding-model-id', default='text-embedding-004')
    parser.add_argument('--gemini-api-key', default=None, help='Optional Gemini API key override. Defaults to GEMINI_API_KEY env var.')
    parser.add_argument('--gemini-base-url', default='https://generativelanguage.googleapis.com/v1beta/openai')
    parser.add_argument('--gemini-timeout-seconds', type=float, default=120.0)
    parser.add_argument('--gemini-embedding-dimensions', type=int, default=None)

    parser.add_argument('--deepseek-embedding-model-id', default='deepseek-embedding')
    parser.add_argument('--deepseek-api-key', default=None, help='Optional DeepSeek API key override. Defaults to DEEPSEEK_API_KEY env var.')
    parser.add_argument('--deepseek-base-url', default='https://api.deepseek.com/v1')
    parser.add_argument('--deepseek-timeout-seconds', type=float, default=120.0)
    parser.add_argument('--deepseek-embedding-dimensions', type=int, default=None)

    parser.add_argument('--chunk-size', type=int, default=140)
    parser.add_argument('--chunk-overlap', type=int, default=24)
    parser.add_argument('--allow-hash-fallback', action='store_true', help='Allow E5 backend to fall back to hashing embedder if E5 model cannot load.')
    parser.add_argument('--metadata-filename', default='metadata.jsonl', help='Metadata output filename (.jsonl recommended for large corpora).')
    parser.add_argument('--corpus-version', default='kankor-corpus@local')
    return parser.parse_args()


def _required_api_key(cli_value: str | None, env_var: str, cli_flag: str) -> str:
    resolved = (cli_value or os.getenv(env_var) or '').strip()
    if resolved:
        return resolved
    raise SystemExit(f'Missing API key for selected embedding backend. Set {cli_flag} or {env_var}.')


def _build_embedder(args: argparse.Namespace) -> tuple[Embedder, str]:
    if args.embedding_backend == 'hash':
        return HashingEmbedder(), 'hash'

    if args.embedding_backend == 'e5':
        return (
            MultilingualE5Embedder(
                model_name=args.embedding_model_id,
                allow_hash_fallback=args.allow_hash_fallback,
            ),
            args.embedding_model_id,
        )

    if args.embedding_backend == 'openai':
        return (
            OpenAIEmbedder(
                model_name=args.openai_embedding_model_id,
                api_key=args.openai_api_key,
                base_url=args.openai_base_url,
                dimensions=args.openai_embedding_dimensions,
                timeout_seconds=args.openai_timeout_seconds,
            ),
            args.openai_embedding_model_id,
        )

    if args.embedding_backend == 'gemini':
        return (
            GeminiEmbedder(
                model_name=args.gemini_embedding_model_id,
                api_key=_required_api_key(args.gemini_api_key, 'GEMINI_API_KEY', '--gemini-api-key'),
                base_url=args.gemini_base_url,
                dimensions=args.gemini_embedding_dimensions,
                timeout_seconds=args.gemini_timeout_seconds,
            ),
            args.gemini_embedding_model_id,
        )

    return (
        DeepSeekEmbedder(
            model_name=args.deepseek_embedding_model_id,
            api_key=_required_api_key(args.deepseek_api_key, 'DEEPSEEK_API_KEY', '--deepseek-api-key'),
            base_url=args.deepseek_base_url,
            dimensions=args.deepseek_embedding_dimensions,
            timeout_seconds=args.deepseek_timeout_seconds,
        ),
        args.deepseek_embedding_model_id,
    )


def embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError('--embedding-batch-size must be > 0')
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        vectors.append(embedder.embed_documents(texts[start:end]))
        print(f'embedded chunks {end}/{len(texts)}')
    return np.vstack(vectors).astype(np.float32)


def main() -> None:
    args = parse_args()
    if not args.input and not args.dataset_name:
        raise SystemExit('Provide either --input or --dataset-name.')

    source = HFDatasetCorpusSource(dataset_name=args.dataset_name, split=args.split, local_path=args.input)
    documents = list(source.load_documents())
    if not documents:
        raise SystemExit('No documents were loaded from the selected corpus source.')

    chunked = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunked:
        raise SystemExit('No chunks were produced. Adjust chunk settings or verify corpus text content.')

    embedder, selected_embedding_model = _build_embedder(args)
    embeddings = embed_in_batches(embedder, [doc.text for doc in chunked], args.embedding_batch_size)
    vector_store = FaissVectorStore.build(embeddings, chunked)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_filename = args.metadata_filename
    vector_store.save(output_dir / 'index.faiss', output_dir / metadata_filename)

    manifest = {
        'documents': len(documents),
        'chunks': len(chunked),
        'embedding_backend': args.embedding_backend,
        'embedding_model_id': selected_embedding_model,
        'embedding_batch_size': args.embedding_batch_size,
        'metadata_filename': metadata_filename,
        'corpus_version': args.corpus_version,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()



