from __future__ import annotations

from dataclasses import dataclass

import httpx

from .contracts import OutboundMessenger


@dataclass(slots=True, frozen=True)
class SentTextMessage:
    to: str
    text: str
    in_reply_to_message_id: str | None = None


class LoggingMessenger(OutboundMessenger):
    def __init__(self) -> None:
        self.sent_messages: list[SentTextMessage] = []

    async def send_text(
        self,
        *,
        to: str,
        text: str,
        in_reply_to_message_id: str | None = None,
    ) -> None:
        self.sent_messages.append(
            SentTextMessage(
                to=to,
                text=text,
                in_reply_to_message_id=in_reply_to_message_id,
            )
        )


class MetaWhatsAppMessenger(OutboundMessenger):
    def __init__(
        self,
        *,
        access_token: str,
        phone_number_id: str,
        api_version: str = "v22.0",
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.access_token = access_token.strip()
        self.phone_number_id = phone_number_id.strip()
        self.api_version = api_version.strip() or "v22.0"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def send_text(
        self,
        *,
        to: str,
        text: str,
        in_reply_to_message_id: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": text,
            },
        }
        if in_reply_to_message_id:
            payload["context"] = {"message_id": in_reply_to_message_id}

        response = await self._client.post(
            f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(
                f"WhatsApp outbound send failed ({response.status_code}): {detail}"
            )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
