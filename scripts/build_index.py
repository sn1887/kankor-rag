from __future__ import annotations
import argparse, json
from pathlib import Path
from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.vector_faiss import FaissVectorStore
from rag_core.util.text_splitter import chunk_documents

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a FAISS index from a JSONL corpus or HF dataset.')
    parser.add_argument('--input', help='Path to a local JSONL corpus file.', default=None)
    parser.add_argument('--dataset-name', help='Optional HF dataset repo id.', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--embedding-model-id', default='intfloat/multilingual-e5-small')
    parser.add_argument('--embedding-backend', choices=['e5', 'hash'], default='e5')
    parser.add_argument('--chunk-size', type=int, default=140)
    parser.add_argument('--chunk-overlap', type=int, default=24)
    parser.add_argument('--corpus-version', default='kankor-corpus@local')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not args.input and not args.dataset_name:
        raise SystemExit('Provide either --input or --dataset-name.')
    source = HFDatasetCorpusSource(dataset_name=args.dataset_name, split=args.split, local_path=args.input)
    documents = list(source.load_documents())
    chunked = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedder = HashingEmbedder() if args.embedding_backend == 'hash' else MultilingualE5Embedder(model_name=args.embedding_model_id)
    embeddings = embedder.embed_documents([doc.text for doc in chunked])
    vector_store = FaissVectorStore.build(embeddings, chunked)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save(output_dir / 'index.faiss', output_dir / 'metadata.json')
    manifest = {'documents': len(documents), 'chunks': len(chunked), 'embedding_backend': args.embedding_backend, 'embedding_model_id': args.embedding_model_id, 'corpus_version': args.corpus_version}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
