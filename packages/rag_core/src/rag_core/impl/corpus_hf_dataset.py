from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

from rag_core.contracts.corpus import CorpusSource
from rag_core.types import Document


class HFDatasetCorpusSource(CorpusSource):
    def __init__(self, *, dataset_name: str | None = None, split: str = 'train', local_path: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.local_path = local_path

    def load_documents(self) -> Iterator[Document]:
        if self.local_path:
            path = Path(self.local_path)
            with path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    row = json.loads(raw)
                    yield Document(id=row['id'], text=row['text'], metadata=row.get('metadata', {}))
            return
        if not self.dataset_name:
            raise ValueError('Either dataset_name or local_path must be provided.')
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, split=self.split)
        for row in dataset:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            yield Document(id=str(row['id']), text=text, metadata=row.get('metadata', {}))
