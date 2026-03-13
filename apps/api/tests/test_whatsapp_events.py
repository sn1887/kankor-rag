from __future__ import annotations

from kankor_api.whatsapp.events import parse_inbound_messages


def test_parse_inbound_messages_extracts_supported_and_unsupported_messages() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": "12345"},
                            "contacts": [
                                {"wa_id": "93700111222", "profile": {"name": "Ahmad"}}
                            ],
                            "messages": [
                                {
                                    "id": "wamid-1",
                                    "from": "93700111222",
                                    "timestamp": "1711000000",
                                    "type": "text",
                                    "text": {"body": "سلام، سوال ریاضی دارم"},
                                },
                                {
                                    "id": "wamid-2",
                                    "from": "93700111222",
                                    "timestamp": "1711000001",
                                    "type": "image",
                                    "image": {
                                        "id": "media-123",
                                        "mime_type": "image/jpeg",
                                        "caption": "لطفا این سوال را حل کنید",
                                        "sha256": "deadbeef",
                                    },
                                },
                                {
                                    "id": "wamid-3",
                                    "from": "93700111222",
                                    "timestamp": "1711000002",
                                    "type": "audio",
                                },
                            ],
                        }
                    }
                ]
            }
        ]
    }

    messages = parse_inbound_messages(payload)

    assert len(messages) == 3

    text_message = messages[0]
    assert text_message.message_id == "wamid-1"
    assert text_message.message_type == "text"
    assert text_message.text is not None
    assert text_message.text.body == "سلام، سوال ریاضی دارم"
    assert text_message.contact_name == "Ahmad"
    assert text_message.phone_number_id == "12345"
    assert text_message.is_supported is True

    image_message = messages[1]
    assert image_message.message_type == "image"
    assert image_message.image is not None
    assert image_message.image.media_id == "media-123"
    assert image_message.image.caption == "لطفا این سوال را حل کنید"
    assert image_message.is_supported is True

    unsupported_message = messages[2]
    assert unsupported_message.message_type == "audio"
    assert unsupported_message.is_supported is False
