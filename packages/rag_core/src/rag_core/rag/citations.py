from __future__ import annotations
from collections.abc import Sequence
from rag_core.types import Hit

def truncate(text: str, width: int = 280) -> str:
    stripped = ' '.join(text.split())
    return stripped if len(stripped) <= width else stripped[: width - 1].rstrip() + '…'

def hits_to_source_payload(hits: Sequence[Hit], corpus_version: str) -> list[dict]:
    payload: list[dict] = []
    for idx, hit in enumerate(hits, start=1):
        meta = hit.document.metadata
        payload.append({'id': hit.document.id, 'badge': f'S{idx}', 'title': meta.get('title') or meta.get('subject') or hit.document.id, 'snippet': truncate(hit.document.text), 'score': float(hit.score), 'subject': str(meta.get('subject', 'unknown')), 'language': str(meta.get('language', 'unknown')), 'gradeBand': str(meta.get('grade_band', 'mixed')), 'corpusVersion': corpus_version})
    return payload
