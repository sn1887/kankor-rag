from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from rag_core.contracts.embeddings import Embedder
from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.embeddings_openai import OpenAIEmbedder
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.util.text_splitter import chunk_documents

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a FAISS index from a JSONL corpus or HF dataset.')
    parser.add_argument('--input', help='Path to a local JSONL corpus file.', default=None)
    parser.add_argument('--dataset-name', help='Optional HF dataset repo id.', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--embedding-model-id', default='intfloat/multilingual-e5-small')
    parser.add_argument('--openai-embedding-model-id', default='text-embedding-3-small')
    parser.add_argument('--embedding-backend', choices=['e5', 'hash', 'openai'], default='e5')
    parser.add_argument('--openai-api-key', default=None, help='Optional OpenAI API key override. Defaults to OPENAI_API_KEY env var.')
    parser.add_argument('--openai-base-url', default=None, help='Optional OpenAI-compatible base URL.')
    parser.add_argument('--openai-timeout-seconds', type=float, default=120.0)
    parser.add_argument('--openai-embedding-dimensions', type=int, default=None)
    parser.add_argument('--embedding-batch-size', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, default=140)
    parser.add_argument('--chunk-overlap', type=int, default=24)
    parser.add_argument('--allow-hash-fallback', action='store_true', help='Allow E5 backend to fall back to hashing embedder if E5 model cannot load.')
    parser.add_argument('--metadata-filename', default='metadata.jsonl', help='Metadata output filename (.jsonl recommended for large corpora).')
    parser.add_argument('--corpus-version', default='kankor-corpus@local')
    return parser.parse_args()

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
    if args.embedding_backend == 'hash':
        embedder = HashingEmbedder()
        selected_embedding_model = 'hash'
    elif args.embedding_backend == 'e5':
        embedder = MultilingualE5Embedder(
            model_name=args.embedding_model_id,
            allow_hash_fallback=args.allow_hash_fallback,
        )
        selected_embedding_model = args.embedding_model_id
    else:
        embedder = OpenAIEmbedder(
            model_name=args.openai_embedding_model_id,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            dimensions=args.openai_embedding_dimensions,
            timeout_seconds=args.openai_timeout_seconds,
        )
        selected_embedding_model = args.openai_embedding_model_id
    embeddings = embed_in_batches(embedder, [doc.text for doc in chunked], args.embedding_batch_size)
    vector_store = FaissVectorStore.build(embeddings, chunked)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_filename = args.metadata_filename
    vector_store.save(output_dir / 'index.faiss', output_dir / metadata_filename)
    manifest = {'documents': len(documents), 'chunks': len(chunked), 'embedding_backend': args.embedding_backend, 'embedding_model_id': selected_embedding_model, 'embedding_batch_size': args.embedding_batch_size, 'metadata_filename': metadata_filename, 'corpus_version': args.corpus_version}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
