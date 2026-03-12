from __future__ import annotations

import pytest
from fastapi import HTTPException

from kankor_api.auth import require_bearer_api_key


def test_require_bearer_api_key_accepts_valid_token() -> None:
    require_bearer_api_key(
        authorization="Bearer my-token",
        expected_token="my-token",
        endpoint_label="test-endpoint",
    )


def test_require_bearer_api_key_rejects_invalid_token() -> None:
    with pytest.raises(HTTPException) as exc:
        require_bearer_api_key(
            authorization="Bearer wrong-token",
            expected_token="my-token",
            endpoint_label="test-endpoint",
        )
    assert exc.value.status_code == 401
