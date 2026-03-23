from __future__ import annotations

import httpx

from .contracts import MediaBlob, MediaProvider


class MediaFetchError(RuntimeError):
    pass


class NoopMediaProvider(MediaProvider):
    async def fetch(self, *, media_id: str, hinted_mime_type: str | None = None) -> MediaBlob:
        raise MediaFetchError(
            "Media fetching is disabled. Configure a WhatsApp media provider backend."
        )


class MetaWhatsAppMediaProvider(MediaProvider):
    def __init__(
        self,
        *,
        access_token: str,
        api_version: str = "v22.0",
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.access_token = access_token.strip()
        self.api_version = api_version.strip() or "v22.0"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def fetch(self, *, media_id: str, hinted_mime_type: str | None = None) -> MediaBlob:
        metadata = await self._get_media_metadata(media_id)
        media_url = str(metadata.get("url") or "").strip()
        if not media_url:
            raise MediaFetchError("Meta media lookup did not return a media URL.")

        response = await self._client.get(
            media_url,
            headers={"Authorization": f"Bearer {self.access_token}"},
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise MediaFetchError(
                f"Meta media download failed ({response.status_code}): {detail}"
            )

        resolved_mime_type = str(metadata.get("mime_type") or "").strip() or hinted_mime_type
        resolved_sha = str(metadata.get("sha256") or "").strip() or None
        return MediaBlob(content=response.content, mime_type=resolved_mime_type, sha256=resolved_sha)

    async def _get_media_metadata(self, media_id: str) -> dict:
        response = await self._client.get(
            f"https://graph.facebook.com/{self.api_version}/{media_id}",
            headers={"Authorization": f"Bearer {self.access_token}"},
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise MediaFetchError(
                f"Meta media metadata request failed ({response.status_code}): {detail}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise MediaFetchError("Meta media metadata response was not a JSON object.")
        return payload

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
