#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import time
import uuid

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Kankor WhatsApp webhook verification and ingestion endpoints.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--verify-token", required=True, help="Expected verify token configured in API")
    parser.add_argument(
        "--webhook-secret",
        default="",
        help="Webhook secret to auto-generate X-Hub-Signature-256",
    )
    parser.add_argument("--from-wa-id", default="93700000000", help="Synthetic WhatsApp sender ID")
    parser.add_argument("--phone-number-id", default="test-phone-id", help="Synthetic phone number id")
    parser.add_argument("--message", default="سلام! لطفا سوال کَنکور را توضیح بده.", help="Text message payload")
    parser.add_argument("--message-id", default="", help="Optional override for WhatsApp message id")
    parser.add_argument("--challenge", default="", help="Optional challenge string")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")
    return parser.parse_args()


def build_payload(
    *,
    message_id: str,
    from_wa_id: str,
    phone_number_id: str,
    text: str,
) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "smoke-entry",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": phone_number_id},
                            "contacts": [
                                {
                                    "wa_id": from_wa_id,
                                    "profile": {"name": "Smoke Test User"},
                                }
                            ],
                            "messages": [
                                {
                                    "id": message_id,
                                    "from": from_wa_id,
                                    "timestamp": str(int(time.time())),
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def build_signature(secret: str, payload_bytes: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def print_response(label: str, response: httpx.Response) -> None:
    print(f"\n[{label}] status={response.status_code}")
    body = response.text.strip()
    if body:
        print(body)


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    webhook_url = f"{base_url}/v1/whatsapp/webhook"

    challenge = args.challenge.strip() or f"smoke-{int(time.time())}"
    message_id = args.message_id.strip() or f"wamid.smoke.{uuid.uuid4().hex[:12]}"

    payload = build_payload(
        message_id=message_id,
        from_wa_id=args.from_wa_id.strip(),
        phone_number_id=args.phone_number_id.strip(),
        text=args.message,
    )
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    with httpx.Client(timeout=args.timeout) as client:
        verify_response = client.get(
            webhook_url,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": args.verify_token,
                "hub.challenge": challenge,
            },
        )
        print_response("verify", verify_response)

        headers = {"Content-Type": "application/json"}
        if args.webhook_secret.strip():
            headers["X-Hub-Signature-256"] = build_signature(args.webhook_secret.strip(), payload_bytes)

        ingest_response = client.post(webhook_url, content=payload_bytes, headers=headers)
        print_response("ingest", ingest_response)

    verify_ok = verify_response.status_code == 200 and verify_response.text.strip() == challenge
    ingest_ok = 200 <= ingest_response.status_code < 300
    if verify_ok and ingest_ok:
        print("\nSmoke test passed.")
        return 0

    print("\nSmoke test failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
