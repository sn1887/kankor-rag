from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SUPPORTED_WHATSAPP_MESSAGE_TYPES = {"text", "image"}


@dataclass(slots=True, frozen=True)
class WhatsAppTextPayload:
    body: str


@dataclass(slots=True, frozen=True)
class WhatsAppImagePayload:
    media_id: str
    mime_type: str | None = None
    sha256: str | None = None
    caption: str | None = None


@dataclass(slots=True, frozen=True)
class WhatsAppInboundMessage:
    message_id: str
    from_wa_id: str
    timestamp: int | None
    message_type: str
    contact_name: str | None = None
    phone_number_id: str | None = None
    text: WhatsAppTextPayload | None = None
    image: WhatsAppImagePayload | None = None

    @property
    def is_supported(self) -> bool:
        if self.message_type == "text":
            return self.text is not None
        if self.message_type == "image":
            return self.image is not None
        return False


def parse_inbound_messages(payload: Mapping[str, Any]) -> list[WhatsAppInboundMessage]:
    messages: list[WhatsAppInboundMessage] = []
    for entry in _as_list(payload.get("entry")):
        for change in _as_list(_as_mapping(entry).get("changes")):
            value = _as_mapping(_as_mapping(change).get("value"))
            phone_number_id = _as_str(_as_mapping(value.get("metadata")).get("phone_number_id"))
            contacts = _build_contacts_lookup(value)
            for raw_message in _as_list(value.get("messages")):
                normalized = _parse_message(
                    raw_message=_as_mapping(raw_message),
                    contacts=contacts,
                    phone_number_id=phone_number_id,
                )
                if normalized is not None:
                    messages.append(normalized)
    return messages


def _parse_message(
    *,
    raw_message: Mapping[str, Any],
    contacts: Mapping[str, str | None],
    phone_number_id: str | None,
) -> WhatsAppInboundMessage | None:
    message_id = _as_str(raw_message.get("id"))
    from_wa_id = _as_str(raw_message.get("from"))
    message_type = _as_str(raw_message.get("type")).lower()
    if not message_id or not from_wa_id or not message_type:
        return None

    text_payload: WhatsAppTextPayload | None = None
    image_payload: WhatsAppImagePayload | None = None

    if message_type == "text":
        body = _as_str(_as_mapping(raw_message.get("text")).get("body"))
        text_payload = WhatsAppTextPayload(body=body)
    elif message_type == "image":
        image = _as_mapping(raw_message.get("image"))
        media_id = _as_str(image.get("id"))
        if media_id:
            image_payload = WhatsAppImagePayload(
                media_id=media_id,
                mime_type=_nullable_str(image.get("mime_type")),
                sha256=_nullable_str(image.get("sha256")),
                caption=_nullable_str(image.get("caption")),
            )

    return WhatsAppInboundMessage(
        message_id=message_id,
        from_wa_id=from_wa_id,
        timestamp=_parse_timestamp(raw_message.get("timestamp")),
        message_type=message_type,
        contact_name=contacts.get(from_wa_id),
        phone_number_id=phone_number_id,
        text=text_payload,
        image=image_payload,
    )


def _build_contacts_lookup(value: Mapping[str, Any]) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for contact in _as_list(value.get("contacts")):
        normalized = _as_mapping(contact)
        wa_id = _as_str(normalized.get("wa_id"))
        if not wa_id:
            continue
        profile_name = _nullable_str(_as_mapping(normalized.get("profile")).get("name"))
        lookup[wa_id] = profile_name
    return lookup


def _parse_timestamp(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    return {}


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    return []


def _as_str(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _nullable_str(raw: Any) -> str | None:
    value = _as_str(raw)
    return value or None
