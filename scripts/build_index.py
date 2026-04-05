from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_core.contracts.embeddings import Embedder
from rag_core.impl.corpus_pages import CanonicalPageCorpusSource, StructureIndexBuilder
from rag_core.impl.corpus_hf_dataset import HFDatasetCorpusSource
from rag_core.impl.embeddings_bge_m3 import BGEM3Embedder
from rag_core.impl.embeddings_deepseek import DeepSeekEmbedder
from rag_core.impl.embeddings_e5 import HashingEmbedder, MultilingualE5Embedder
from rag_core.impl.embeddings_gemini import GeminiEmbedder
from rag_core.impl.embeddings_openai import OpenAIEmbedder
from rag_core.impl.faiss_streaming import StreamingFaissArtifactWriter
from rag_core.types import Document
from rag_core.util.text_splitter import iter_chunked_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a FAISS index from a JSONL corpus or HF dataset.')
    parser.add_argument('--input', help='Path to a local JSONL corpus file.', default=None)
    parser.add_argument('--corpus-root', help='Path to the canonical new corpus root.', default=None)
    parser.add_argument('--frontmatter-toc-path', help='Path to normalized frontmatter TOC jsonl.', default=None)
    parser.add_argument('--dataset-name', help='Optional HF dataset repo id.', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--embedding-backend', choices=['e5', 'hash', 'bge_m3', 'openai', 'gemini', 'deepseek'], default='e5')
    parser.add_argument('--embedding-model-id', default='intfloat/multilingual-e5-small')
    parser.add_argument('--embedding-batch-size', type=int, default=128)
    parser.add_argument('--bge-m3-model-id', default='BAAI/bge-m3')
    parser.add_argument('--bge-m3-batch-size', type=int, default=32)
    parser.add_argument('--bge-m3-use-fp16', action='store_true')
    parser.add_argument('--bge-m3-device', default=None)
    parser.add_argument('--bge-m3-max-length', type=int, default=8192)

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

    if args.embedding_backend == 'bge_m3':
        return (
            BGEM3Embedder(
                model_name=args.bge_m3_model_id,
                batch_size=args.bge_m3_batch_size,
                use_fp16=args.bge_m3_use_fp16,
                device=args.bge_m3_device,
                max_length=args.bge_m3_max_length,
            ),
            args.bge_m3_model_id,
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


def _flush_batch(
    *,
    embedder: Embedder,
    writer: StreamingFaissArtifactWriter,
    documents: list[Document],
    embedded_so_far: int,
) -> tuple[list[Document], int]:
    if not documents:
        return documents, embedded_so_far
    vectors = embedder.embed_documents([document.text for document in documents])
    writer.add_batch(embeddings=vectors, documents=documents)
    embedded = embedded_so_far + len(documents)
    print(f'embedded chunks {embedded}')
    return [], embedded


def _write_document_jsonl(path: Path, documents: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "id": document.id,
                        "text": document.text,
                        "metadata": document.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    if not args.input and not args.dataset_name and not args.corpus_root:
        raise SystemExit('Provide one of --corpus-root, --input, or --dataset-name.')
    if args.embedding_batch_size <= 0:
        raise SystemExit('--embedding-batch-size must be > 0')

    if args.corpus_root:
        source = CanonicalPageCorpusSource(
            corpus_root=args.corpus_root,
            corpus_version=args.corpus_version,
            frontmatter_toc_path=args.frontmatter_toc_path,
        )
    else:
        source = HFDatasetCorpusSource(dataset_name=args.dataset_name, split=args.split, local_path=args.input)
    embedder, selected_embedding_model = _build_embedder(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_filename = args.metadata_filename
    index_path = output_dir / 'index.faiss'
    metadata_path = output_dir / metadata_filename
    chapter_index_path = output_dir / "chapter_index.jsonl"
    topic_index_path = output_dir / "topic_index.jsonl"

    documents = 0
    chunks = 0
    embedded_chunks = 0
    pending_batch: list[Document] = []
    embedding_dimension: int | None = None

    with StreamingFaissArtifactWriter(index_path=index_path, metadata_path=metadata_path) as writer:
        for source_document in source.load_documents():
            documents += 1
            chunk_iterable = (
                [source_document]
                if args.corpus_root
                else iter_chunked_documents(
                    [source_document],
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            )
            for chunk in chunk_iterable:
                chunks += 1
                pending_batch.append(chunk)
                if len(pending_batch) >= args.embedding_batch_size:
                    pending_batch, embedded_chunks = _flush_batch(
                        embedder=embedder,
                        writer=writer,
                        documents=pending_batch,
                        embedded_so_far=embedded_chunks,
                    )
                    embedding_dimension = writer.dimension

        pending_batch, embedded_chunks = _flush_batch(
            embedder=embedder,
            writer=writer,
            documents=pending_batch,
            embedded_so_far=embedded_chunks,
        )
        embedding_dimension = writer.dimension

        if documents == 0:
            raise SystemExit('No documents were loaded from the selected corpus source.')
        if chunks == 0:
            raise SystemExit('No chunks were produced. Adjust chunk settings or verify corpus text content.')

        writer.save_index()

    if args.corpus_root:
        structure_builder = StructureIndexBuilder(
            corpus_root=args.corpus_root,
            corpus_version=args.corpus_version,
            frontmatter_toc_path=args.frontmatter_toc_path,
        )
        chapter_documents = structure_builder.build_chapter_documents()
        topic_documents = structure_builder.build_topic_documents()
        _write_document_jsonl(chapter_index_path, chapter_documents)
        _write_document_jsonl(topic_index_path, topic_documents)

    manifest = {
        'documents': documents,
        'chunks': chunks,
        'embedding_backend': args.embedding_backend,
        'embedding_model_id': selected_embedding_model,
        'embedding_dimension': embedding_dimension,
        'embedding_batch_size': args.embedding_batch_size,
        'metadata_filename': metadata_filename,
        'corpus_version': args.corpus_version,
        'chapter_index_filename': chapter_index_path.name if args.corpus_root else None,
        'topic_index_filename': topic_index_path.name if args.corpus_root else None,
        'corpus_format': 'canonical_pages' if args.corpus_root else 'jsonl_or_dataset',
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
