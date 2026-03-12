from __future__ import annotations

from fastapi import HTTPException


def require_bearer_api_key(
    *,
    authorization: str | None,
    expected_token: str | None,
    endpoint_label: str,
) -> None:
    """Validate Bearer API key when endpoint protection is configured."""
    expected = (expected_token or "").strip()
    if not expected:
        return

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if token != expected:
        raise HTTPException(status_code=401, detail=f"Invalid API key for {endpoint_label}")
