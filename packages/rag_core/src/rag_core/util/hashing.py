from __future__ import annotations
import hashlib, json
from typing import Any

def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]
