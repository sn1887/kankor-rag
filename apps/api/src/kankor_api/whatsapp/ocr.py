from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .contracts import MediaBlob, OCRProvider, OCRResult


class NoopOCRProvider(OCRProvider):
    async def extract_text(self, media: MediaBlob) -> OCRResult:
        _ = media
        return OCRResult(text="", confidence=0.0, engine="noop")


class TesseractOCRProvider(OCRProvider):
    def __init__(
        self,
        *,
        language: str = "fas+pus+eng",
        timeout_seconds: float = 20.0,
        command: str = "tesseract",
    ) -> None:
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.command = command

    async def extract_text(self, media: MediaBlob) -> OCRResult:
        return self._extract_sync(media)

    def _extract_sync(self, media: MediaBlob) -> OCRResult:
        suffix = _guess_suffix(media.mime_type)
        with tempfile.TemporaryDirectory(prefix="wa-ocr-") as tmp_dir:
            input_path = Path(tmp_dir) / f"input{suffix}"
            output_base = Path(tmp_dir) / "ocr"
            output_path = output_base.with_suffix(".txt")

            input_path.write_bytes(media.content)
            try:
                result = subprocess.run(
                    [
                        self.command,
                        str(input_path),
                        str(output_base),
                        "-l",
                        self.language,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "tesseract command not found. Install Tesseract OCR or use RAG_WHATSAPP_OCR_BACKEND=noop."
                ) from exc

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                raise RuntimeError(f"tesseract OCR failed: {stderr or 'unknown error'}")

            text = ""
            if output_path.exists():
                text = output_path.read_text(encoding="utf-8", errors="ignore").strip()
            return OCRResult(text=text, confidence=None, engine="tesseract")


def _guess_suffix(mime_type: str | None) -> str:
    normalized = (mime_type or "").strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/tiff": ".tif",
        "image/bmp": ".bmp",
    }
    return mapping.get(normalized, ".img")
