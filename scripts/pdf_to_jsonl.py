#!/usr/bin/env python3
"""
Drop-in replacement PDF -> JSONL ingestion script for Kankor textbooks.

Why this version exists
-----------------------
Many MoE textbooks have an embedded text layer that is technically present but
poorly exposed by pdfminer/pdfplumber: broken reading order, mixed Latin junk,
missing spaces, or garbled glyph mapping. This script changes the extraction
strategy to a cascade that is far more resilient on legacy Afghan textbooks:

    1) Poppler word-level extraction (pdftotext -bbox-layout)  [primary]
    2) PyMuPDF word/block reconstruction                       [fallback]
    3) OCR (Tesseract) on failed / low-quality pages          [last resort]

Key improvements
----------------
- Coordinate-based reading order instead of trusting the PDF's internal order.
- Right-to-left aware line reconstruction.
- Basic multi-column / multi-block recovery by detecting column structure.
- Stronger garble detection for Arabic-script books polluted with Latin junk.
- Unicode cleanup that preserves real Dari/Pashto text while removing control,
  private-use, bidi, replacement, and other problematic code points.
- Page-level audit trail for debugging difficult books.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterator, Sequence

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pdfplumber  # late fallback only
except Exception:
    pdfplumber = None


SUBJECT_MAP: dict[str, tuple[str, str]] = {
    "math": ("math", "mathematics"),
    "physic": ("natural_science", "physics"),
    "chemistry": ("natural_science", "chemistry"),
    "biology": ("natural_science", "biology"),
    "computer": ("natural_science", "computer_science"),
    "history": ("social_science", "history"),
    "geography": ("social_science", "geography"),
    "civic": ("social_science", "civic_education"),
    "islamic": ("social_science", "islamic_studies"),
    "tafseer": ("social_science", "tafseer"),
    "english": ("languages", "english"),
    "dari": ("languages", "dari"),
    "pashto": ("languages", "pashto"),
}

LANG_MAP: dict[str, str] = {
    "dr": "fa",
    "ps": "ps",
    "ar": "ar",
    "en": "en",
}

OCR_DEFAULT_LANGS = "eng+ara+fas+pus"
OCR_LANG_ALIASES: dict[str, str] = {
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "ar": "ara",
    "ara": "ara",
    "arabic": "ara",
    "fa": "fas",
    "fas": "fas",
    "farsi": "fas",
    "persian": "fas",
    "dr": "fas",
    "dari": "fas",
    "ps": "pus",
    "pus": "pus",
    "pashto": "pus",
}

CHAR_NORMALIZATION_MAP = str.maketrans({
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ة": "ه",
    "ۀ": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
    "آ": "آ",
    "ـ": "",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

BIDI_CONTROL_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_./+-]{0,12}\b")
SHORT_LATIN_JUNK_RE = re.compile(r"\b[A-Za-z]{1,5}\b")
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

LATIN_KEEP_TOKENS = {
    "dna", "rna", "atp", "adp", "ph",
    "cm", "mm", "km", "kg", "mg", "ml",
    "o2", "co2", "h2o", "na", "cl", "ca", "fe", "cu",
    "tv", "it", "pc",
}

METHOD_BONUS = {
    "poppler_bbox": 0.04,
    "pymupdf_words": 0.03,
    "poppler_plain": 0.00,
    "pdfplumber_plain": -0.03,
    "ocr": 0.00,
}


def infer_metadata(pdf_path: Path) -> dict:
    stem = pdf_path.stem
    match = re.match(r"^G(?P<grade>\d+)-(?P<lang>[A-Za-z]{2})-(?P<subject>.+)$", stem)
    if match:
        grade_band = match.group("grade")
        lang_code = LANG_MAP.get(match.group("lang").lower(), "fa")
        subj_raw = match.group("subject").lower()
    else:
        parts = stem.split("-")
        grade_band = parts[0].lstrip("G") if parts else "unknown"
        lang_code = LANG_MAP.get(parts[1].lower(), "fa") if len(parts) > 1 else "fa"
        subj_raw = "-".join(parts[2:]).lower() if len(parts) > 2 else "unknown"

    subj_raw = subj_raw.replace("_", "-")
    subject_category, subject = "social_science", subj_raw
    for key, (cat, subj) in SUBJECT_MAP.items():
        if key in subj_raw:
            subject_category, subject = cat, subj
            break

    return {
        "title": pdf_path.stem,
        "subject_category": subject_category,
        "subject": subject,
        "grade_band": grade_band,
        "language": lang_code,
        "source_type": "explanation",
        "source_id": pdf_path.stem,
        "copyright": "Afghanistan Ministry of Education",
        "license": "Public Domain",
        "retrieval_weight": 1.0,
    }


def normalize_ocr_langs(raw_langs: str) -> str:
    tokens = [item.strip().lower() for item in raw_langs.split("+") if item.strip()]
    normalized = [OCR_LANG_ALIASES.get(token, token) for token in tokens]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in normalized:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return "+".join(deduped) if deduped else OCR_DEFAULT_LANGS


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = html.unescape(text)
    text = text.translate(CHAR_NORMALIZATION_MAP)
    text = re.sub(r"\(cid:\d+\)", " ", text, flags=re.IGNORECASE)
    text = PRIVATE_USE_RE.sub("", text)
    text = BIDI_CONTROL_RE.sub("", text)
    text = text.replace("\ufeff", " ").replace("\u00ad", "")
    text = CONTROL_RE.sub("", text)
    text = re.sub(r"[\u2028\u2029]", "\n", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _script_letter_count(text: str) -> int:
    count = 0
    for ch in text:
        code = ord(ch)
        if (
            ("a" <= ch.lower() <= "z")
            or (0x0600 <= code <= 0x06FF)
            or (0x0750 <= code <= 0x077F)
            or (0x08A0 <= code <= 0x08FF)
            or (0xFB50 <= code <= 0xFDFF)
            or (0xFE70 <= code <= 0xFEFF)
        ):
            count += 1
    return count


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(ARABIC_CHAR_RE.findall(text)) / max(len(text), 1)


def strip_suspicious_latin_noise(text: str, lang: str) -> str:
    if lang not in {"fa", "ps", "ar"}:
        return text
    if arabic_ratio(text) < 0.12:
        return text

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.lower() in LATIN_KEEP_TOKENS:
            return token
        if len(token) <= 5:
            return " "
        return token

    cleaned = SHORT_LATIN_JUNK_RE.sub(repl, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def text_quality_score(text: str, *, lang: str = "fa") -> float:
    if not text:
        return 0.0

    length = len(text)
    letter_count = _script_letter_count(text)
    cid_hits = len(re.findall(r"\bcid\b", text, flags=re.IGNORECASE))
    replacement_hits = text.count("\ufffd")
    latin_tokens = LATIN_TOKEN_RE.findall(text)
    short_latin = [tok for tok in latin_tokens if len(tok) <= 5 and tok.lower() not in LATIN_KEEP_TOKENS]
    private_hits = len(PRIVATE_USE_RE.findall(text))
    bidi_hits = len(BIDI_CONTROL_RE.findall(text))

    letter_ratio = letter_count / max(length, 1)
    cid_ratio = cid_hits / max(len(text.split()), 1)
    replacement_ratio = replacement_hits / max(length, 1)
    short_latin_ratio = len(short_latin) / max(len(text.split()), 1)
    private_ratio = private_hits / max(length, 1)
    bidi_ratio = bidi_hits / max(length, 1)

    score = letter_ratio
    score -= min(0.55, cid_ratio * 2.5)
    score -= min(0.45, replacement_ratio * 10.0)
    score -= min(0.30, private_ratio * 8.0)
    score -= min(0.10, bidi_ratio * 6.0)

    if lang in {"fa", "ps", "ar"}:
        score += min(0.20, arabic_ratio(text) * 0.4)
        score -= min(0.40, short_latin_ratio * 4.0)

    if length < 80:
        score -= 0.12
    return score


@dataclass
class WordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class LineBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    rtl: bool

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _strip_xml_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _safe_fromstring(xml_text: str) -> ET.Element | None:
    try:
        xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE)
        return ET.fromstring(xml_text)
    except Exception:
        return None


def _iter_word_nodes(root: ET.Element) -> Iterator[ET.Element]:
    for node in root.iter():
        if _strip_xml_namespace(node.tag) == "word":
            yield node


def extract_words_poppler_bbox(pdf_path: Path, page_num: int) -> tuple[list[WordBox], float, float]:
    if not command_exists("pdftotext"):
        return [], 0.0, 0.0

    cmd = ["pdftotext", "-bbox-layout", "-f", str(page_num), "-l", str(page_num), str(pdf_path), "-"]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return [], 0.0, 0.0

    xml_text = result.stdout.decode("utf-8", errors="ignore")
    root = _safe_fromstring(xml_text)
    if root is None:
        return [], 0.0, 0.0

    page_width = 0.0
    page_height = 0.0
    for node in root.iter():
        if _strip_xml_namespace(node.tag) == "page":
            page_width = float(node.attrib.get("width", 0.0))
            page_height = float(node.attrib.get("height", 0.0))
            break

    words: list[WordBox] = []
    for node in _iter_word_nodes(root):
        raw_text = clean_text("".join(node.itertext()))
        if not raw_text:
            continue
        try:
            words.append(
                WordBox(
                    text=raw_text,
                    x0=float(node.attrib.get("xMin", 0.0)),
                    y0=float(node.attrib.get("yMin", 0.0)),
                    x1=float(node.attrib.get("xMax", 0.0)),
                    y1=float(node.attrib.get("yMax", 0.0)),
                )
            )
        except Exception:
            continue
    return words, page_width, page_height


def extract_text_poppler_plain(pdf_path: Path, page_num: int) -> str:
    if not command_exists("pdftotext"):
        return ""
    cmd = ["pdftotext", "-enc", "UTF-8", "-f", str(page_num), "-l", str(page_num), str(pdf_path), "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return clean_text(result.stdout)


def extract_words_pymupdf(pdf_path: Path, page_num: int) -> tuple[list[WordBox], float, float]:
    if fitz is None:
        return [], 0.0, 0.0
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            raw_words = page.get_text("words", sort=False)
            words = []
            for item in raw_words:
                if len(item) < 5:
                    continue
                x0, y0, x1, y1, text = item[:5]
                text = clean_text(str(text))
                if text:
                    words.append(WordBox(text=text, x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)))
            return words, page_width, page_height
    except Exception:
        return [], 0.0, 0.0


def extract_text_pdfplumber_plain(pdf_path: Path, page_num: int) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            candidates: list[str] = []
            for layout in (False, True):
                raw = page.extract_text(layout=layout, x_tolerance=2, y_tolerance=2)
                if raw:
                    cleaned = clean_text(raw)
                    if cleaned:
                        candidates.append(cleaned)
            if not candidates:
                return ""
            return max(candidates, key=len)
    except Exception:
        return ""


def is_rtl_text(text: str) -> bool:
    arabic = len(ARABIC_CHAR_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return arabic >= max(2, latin)


def _join_words(words: Sequence[WordBox], *, rtl: bool) -> str:
    ordered = sorted(words, key=lambda w: w.x0, reverse=rtl)
    if not ordered:
        return ""

    parts: list[str] = []
    prev: WordBox | None = None
    median_height = median([max(1.0, w.height) for w in ordered]) if ordered else 10.0
    base_gap = max(1.5, median_height * 0.18)

    for word in ordered:
        token = word.text.strip()
        if not token:
            continue
        if prev is not None:
            gap = (prev.x0 - word.x1) if rtl else (word.x0 - prev.x1)
            if gap > base_gap * 2.5:
                parts.append("  ")
            else:
                parts.append(" ")
        parts.append(token)
        prev = word

    return clean_text("".join(parts))


def group_words_into_lines(words: Sequence[WordBox]) -> list[LineBox]:
    if not words:
        return []

    heights = [w.height for w in words if w.height > 0]
    y_tol = max(2.5, (median(heights) * 0.45) if heights else 4.0)
    sorted_words = sorted(words, key=lambda w: (w.cy, w.x0))

    lines: list[list[WordBox]] = []
    line_cys: list[float] = []
    for word in sorted_words:
        placed = False
        for idx, line in enumerate(lines):
            if abs(word.cy - line_cys[idx]) <= y_tol:
                line.append(word)
                current = [w.cy for w in line]
                line_cys[idx] = sum(current) / len(current)
                placed = True
                break
        if not placed:
            lines.append([word])
            line_cys.append(word.cy)

    out: list[LineBox] = []
    for line_words in lines:
        if not line_words:
            continue
        sample = " ".join(w.text for w in line_words)
        rtl = is_rtl_text(sample)
        line_text = _join_words(line_words, rtl=rtl)
        if not line_text:
            continue
        out.append(
            LineBox(
                text=line_text,
                x0=min(w.x0 for w in line_words),
                y0=min(w.y0 for w in line_words),
                x1=max(w.x1 for w in line_words),
                y1=max(w.y1 for w in line_words),
                rtl=rtl,
            )
        )
    return sorted(out, key=lambda l: l.y0)


def is_probable_page_number(line: LineBox, page_width: float, page_height: float) -> bool:
    text = line.text.strip()
    if not text:
        return False
    near_bottom = line.y0 > page_height * 0.93
    narrow = line.width < page_width * 0.08
    return bool(
        near_bottom
        and narrow
        and re.fullmatch(r"[\[\](){}\-–— ]*\d+[\[\](){}\-–— ]*", text)
    )


def split_lines_into_bands(lines: Sequence[LineBox]) -> list[list[LineBox]]:
    if not lines:
        return []
    heights = [max(1.0, line.height) for line in lines]
    gap_threshold = max(10.0, median(heights) * 1.8)
    ordered = sorted(lines, key=lambda l: l.y0)
    bands: list[list[LineBox]] = [[ordered[0]]]
    prev = ordered[0]
    for line in ordered[1:]:
        gap = line.y0 - prev.y1
        if gap > gap_threshold:
            bands.append([line])
        else:
            bands[-1].append(line)
        prev = line
    return bands


def _detect_column_cut(lines: Sequence[LineBox], page_width: float) -> float | None:
    narrow = [line for line in lines if line.width < page_width * 0.72]
    if len(narrow) < 6:
        return None
    centers = sorted(line.cx for line in narrow)
    if len(centers) < 6:
        return None
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    if not gaps:
        return None
    biggest_gap, idx = max(gaps, key=lambda item: item[0])
    if biggest_gap < page_width * 0.16:
        return None
    return (centers[idx] + centers[idx + 1]) / 2.0


def order_lines_for_reading(lines: Sequence[LineBox], page_width: float, page_height: float) -> list[LineBox]:
    if not lines:
        return []

    usable = [line for line in lines if not is_probable_page_number(line, page_width, page_height)]
    bands = split_lines_into_bands(usable)
    ordered: list[LineBox] = []

    global_rtl = sum(1 for line in usable if line.rtl) >= max(1, len(usable) // 2)

    for band in bands:
        if len(band) <= 1:
            ordered.extend(sorted(band, key=lambda l: l.y0))
            continue

        cut = _detect_column_cut(band, page_width)
        if cut is None:
            ordered.extend(sorted(band, key=lambda l: l.y0))
            continue

        left = [line for line in band if line.cx <= cut]
        right = [line for line in band if line.cx > cut]

        if not left or not right or min(len(left), len(right)) < 2:
            ordered.extend(sorted(band, key=lambda l: l.y0))
            continue

        first_col = right if global_rtl else left
        second_col = left if global_rtl else right
        ordered.extend(sorted(first_col, key=lambda l: l.y0))
        ordered.extend(sorted(second_col, key=lambda l: l.y0))

    return ordered


def lines_to_page_text(lines: Sequence[LineBox], page_width: float, page_height: float, *, lang: str) -> str:
    ordered = order_lines_for_reading(lines, page_width, page_height)
    if not ordered:
        return ""

    heights = [max(1.0, line.height) for line in ordered]
    median_h = median(heights) if heights else 12.0
    paragraph_gap = max(10.0, median_h * 1.05)

    chunks: list[str] = []
    prev: LineBox | None = None
    for line in ordered:
        text = clean_text(line.text)
        if not text:
            continue
        if prev is None:
            chunks.append(text)
            prev = line
            continue

        gap = line.y0 - prev.y1
        if gap > paragraph_gap:
            chunks.append("\n\n" + text)
        else:
            chunks.append("\n" + text)
        prev = line

    text = "".join(chunks)
    text = clean_text(text)
    text = strip_suspicious_latin_noise(text, lang)
    return text


def _page_image_pymupdf(pdf_path: Path, page_num: int, dpi: int):
    if fitz is None or Image is None:
        return None
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            scale = dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            return Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    except Exception:
        return None


def _extract_page_ocr_text(
    pdf_path: Path,
    page_num: int,
    *,
    ocr_langs: str,
    ocr_dpi: int,
    psm_values: Sequence[int] = (3, 6),
    lang: str = "fa",
) -> str:
    try:
        import pytesseract
    except Exception:
        return ""

    image = _page_image_pymupdf(pdf_path, page_num, ocr_dpi)
    if image is None:
        return ""

    candidates: list[str] = []
    for psm in psm_values:
        try:
            raw = pytesseract.image_to_string(image, lang=ocr_langs, config=f"--oem 1 --psm {psm}")
        except Exception:
            continue
        cleaned = strip_suspicious_latin_noise(clean_text(raw), lang)
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return ""
    return max(candidates, key=lambda item: text_quality_score(item, lang=lang))


def _page_count(pdf_path: Path) -> int:
    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                return len(doc)
        except Exception:
            pass
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception:
            pass
    return 0


def extract_page_best(
    pdf_path: Path,
    page_num: int,
    *,
    lang: str,
    min_quality: float,
    ocr_fallback: bool,
    ocr_primary: bool,
    ocr_langs: str,
    ocr_dpi: int,
) -> tuple[str, float, str, list[dict[str, object]]]:
    candidates: list[tuple[str, str]] = []
    debug: list[dict[str, object]] = []

    if ocr_primary:
        ocr_text = _extract_page_ocr_text(
            pdf_path, page_num, ocr_langs=ocr_langs, ocr_dpi=ocr_dpi, lang=lang
        )
        if ocr_text:
            candidates.append(("ocr", ocr_text))

    poppler_words, pw, ph = extract_words_poppler_bbox(pdf_path, page_num)
    if poppler_words:
        poppler_lines = group_words_into_lines(poppler_words)
        poppler_structured = lines_to_page_text(poppler_lines, pw, ph, lang=lang)
        if poppler_structured:
            candidates.append(("poppler_bbox", poppler_structured))

    poppler_plain = extract_text_poppler_plain(pdf_path, page_num)
    if poppler_plain:
        poppler_plain = strip_suspicious_latin_noise(poppler_plain, lang)
        candidates.append(("poppler_plain", poppler_plain))

    pymu_words, mw, mh = extract_words_pymupdf(pdf_path, page_num)
    if pymu_words:
        pymu_lines = group_words_into_lines(pymu_words)
        pymu_structured = lines_to_page_text(pymu_lines, mw, mh, lang=lang)
        if pymu_structured:
            candidates.append(("pymupdf_words", pymu_structured))

    if pdfplumber is not None:
        plumber_plain = extract_text_pdfplumber_plain(pdf_path, page_num)
        if plumber_plain:
            plumber_plain = strip_suspicious_latin_noise(plumber_plain, lang)
            candidates.append(("pdfplumber_plain", plumber_plain))

    if not ocr_primary and ocr_fallback:
        current_best_score = max(
            (text_quality_score(text, lang=lang) + METHOD_BONUS.get(method, 0.0) for method, text in candidates),
            default=-1.0,
        )
        if current_best_score < max(min_quality, 0.28):
            ocr_text = _extract_page_ocr_text(
                pdf_path, page_num, ocr_langs=ocr_langs, ocr_dpi=ocr_dpi, lang=lang
            )
            if ocr_text:
                candidates.append(("ocr", ocr_text))

    best_method = ""
    best_text = ""
    best_score = -1.0
    seen: set[str] = set()

    for method, text in candidates:
        text = clean_text(text)
        if not text or text in seen:
            continue
        seen.add(text)
        raw_score = text_quality_score(text, lang=lang)
        score = raw_score + METHOD_BONUS.get(method, 0.0)
        debug.append({
            "method": method,
            "score": round(score, 4),
            "raw_score": round(raw_score, 4),
            "length": len(text),
            "preview": text[:220],
        })
        if score > best_score:
            best_method = method
            best_text = text
            best_score = score

    return best_text, best_score, best_method, debug


def is_front_matter_page(text: str, page_num: int, max_front_matter_pages: int) -> bool:
    if page_num > max_front_matter_pages:
        return False
    patterns = [
        r"سرود\s*ملی",
        r"مشخصات\s*کتاب",
        r"پیام\s*وزیر",
        r"وزارت\s*معارف",
        r"فهرست",
        r"پیشگفتار",
        r"حق\s*طبع",
        r"ناشر",
        r"چاپ",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def extract_pages(
    pdf_path: Path,
    min_quality: float = 0.24,
    stats: dict | None = None,
    *,
    language: str,
    ocr_fallback: bool = False,
    ocr_primary: bool = False,
    ocr_langs: str = OCR_DEFAULT_LANGS,
    ocr_dpi: int = 300,
    unreadable_pages: list[dict] | None = None,
    audit_pages: list[dict] | None = None,
) -> Iterator[tuple[int, str]]:
    if stats is None:
        stats = {}
    stats.setdefault("pages_total", 0)
    stats.setdefault("pages_kept", 0)
    stats.setdefault("pages_dropped_quality", 0)
    stats.setdefault("pages_dropped_empty", 0)
    stats.setdefault("pages_ocr_used", 0)

    total = _page_count(pdf_path)
    stats["pages_total"] = total
    if total <= 0:
        print(f"  Warning: Could not determine page count for {pdf_path.name}", file=sys.stderr)
        return

    for i in range(1, total + 1):
        best, best_score, method, debug = extract_page_best(
            pdf_path,
            i,
            lang=language,
            min_quality=min_quality,
            ocr_fallback=ocr_fallback,
            ocr_primary=ocr_primary,
            ocr_langs=ocr_langs,
            ocr_dpi=ocr_dpi,
        )

        if audit_pages is not None:
            audit_pages.append({
                "source_id": pdf_path.stem,
                "page": i,
                "best_method": method,
                "best_score": round(best_score, 4),
                "candidates": debug,
            })

        if not best or len(best) < 40:
            stats["pages_dropped_empty"] += 1
            if unreadable_pages is not None:
                unreadable_pages.append({
                    "source_id": pdf_path.stem,
                    "page": i,
                    "reason": "empty_or_too_short",
                    "score": round(best_score, 4),
                    "method": method,
                    "preview": best[:220] if best else "",
                    "candidates": debug,
                })
            continue

        if best_score < min_quality:
            stats["pages_dropped_quality"] += 1
            if unreadable_pages is not None:
                unreadable_pages.append({
                    "source_id": pdf_path.stem,
                    "page": i,
                    "reason": "low_quality",
                    "score": round(best_score, 4),
                    "method": method,
                    "preview": best[:220],
                    "candidates": debug,
                })
            continue

        if method == "ocr":
            stats["pages_ocr_used"] += 1
        stats["pages_kept"] += 1
        yield i, best


def split_into_paragraphs(
    text: str,
    min_len: int = 80,
    max_len: int = 700,
    *,
    lang: str = "fa",
) -> list[str]:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+([،؛,:!?؟\.])", r"\1", text)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in re.split(r"(?<=[\.\!\?\u061f\u06d4])\s+", text) if p.strip()]

    out: list[str] = []
    for para in paragraphs:
        para = strip_suspicious_latin_noise(clean_text(para), lang)
        if len(para) < min_len:
            continue
        if len(para) <= max_len:
            out.append(para)
            continue
        for i in range(0, len(para), max_len):
            chunk = para[i:i + max_len].strip()
            if len(chunk) >= min_len:
                out.append(chunk)
    return out


def make_record(text: str, metadata: dict, source_id: str, page: int, para_idx: int) -> dict:
    uid = hashlib.sha1(f"{source_id}:p{page}:c{para_idx}:{text[:40]}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": uid,
        "text": text,
        "metadata": {**metadata, "page": page, "chunk_index": para_idx},
    }


REQUIRED_META = {"subject_category", "subject", "grade_band", "language", "source_type"}
VALID_CATS = {"math", "natural_science", "social_science", "languages"}
VALID_LANGS = {"ps", "fa", "ar", "en"}
VALID_TYPES = {"explanation", "definition", "worked_example", "practice_question"}


def validate(record: dict) -> bool:
    m = record.get("metadata", {})
    return (
        bool(record.get("id"))
        and bool(record.get("text"))
        and REQUIRED_META.issubset(m.keys())
        and m["subject_category"] in VALID_CATS
        and m["language"] in VALID_LANGS
        and m["source_type"] in VALID_TYPES
    )


def process_pdf(
    pdf_path: Path,
    corpus_version: str,
    *,
    min_page_quality: float,
    min_paragraph_len: int,
    ocr_fallback: bool,
    ocr_primary: bool,
    ocr_langs: str,
    ocr_dpi: int,
    skip_front_matter: bool,
    max_front_matter_pages: int,
    unreadable_pages: list[dict] | None,
    audit_pages: list[dict] | None,
) -> tuple[list[dict], dict]:
    metadata = infer_metadata(pdf_path)
    metadata["corpus_version"] = corpus_version
    page_stats: dict = {
        "pages_total": 0,
        "pages_kept": 0,
        "pages_dropped_quality": 0,
        "pages_dropped_empty": 0,
        "pages_ocr_used": 0,
        "pages_skipped_front_matter": 0,
    }
    chunks: list[dict] = []

    for page_num, page_text in extract_pages(
        pdf_path,
        min_quality=min_page_quality,
        stats=page_stats,
        language=metadata["language"],
        ocr_fallback=ocr_fallback,
        ocr_primary=ocr_primary,
        ocr_langs=ocr_langs,
        ocr_dpi=ocr_dpi,
        unreadable_pages=unreadable_pages,
        audit_pages=audit_pages,
    ):
        if skip_front_matter and is_front_matter_page(page_text, page_num, max_front_matter_pages):
            page_stats["pages_skipped_front_matter"] += 1
            continue
        paragraphs = split_into_paragraphs(
            page_text,
            min_len=min_paragraph_len,
            lang=metadata["language"],
        )
        for para_idx, para in enumerate(paragraphs):
            record = make_record(
                text=para,
                metadata=metadata,
                source_id=pdf_path.stem,
                page=page_num,
                para_idx=para_idx,
            )
            if validate(record):
                chunks.append(record)

    return chunks, page_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw_pdfs",
                        help="Root folder containing grade_10/, grade_11/, grade_12/")
    parser.add_argument("--output", default="data/corpus/kankor_corpus.jsonl")
    parser.add_argument("--corpus-version", default="kankor-corpus-2026.03")
    parser.add_argument("--min-page-quality", type=float, default=0.24,
                        help="Drop pages with extraction quality below this score.")
    parser.add_argument("--min-paragraph-len", type=int, default=80,
                        help="Drop short chunks below this length.")
    parser.add_argument("--ocr-fallback", action="store_true",
                        help="Run OCR when text extraction is missing or low quality.")
    parser.add_argument("--ocr-primary", action="store_true",
                        help="Use OCR as the primary extractor and text layer as fallback.")
    parser.add_argument("--ocr-langs", default=OCR_DEFAULT_LANGS,
                        help="OCR language codes for tesseract (default: eng+ara+fas+pus).")
    parser.add_argument("--ocr-dpi", type=int, default=300,
                        help="Render DPI used for OCR pages.")
    parser.add_argument("--skip-front-matter", action="store_true", default=True,
                        help="Skip title/preface/legal pages near document start.")
    parser.add_argument("--no-skip-front-matter", action="store_false", dest="skip_front_matter",
                        help="Disable front-matter filtering.")
    parser.add_argument("--max-front-matter-pages", type=int, default=8,
                        help="Apply front-matter filter only within first N pages.")
    parser.add_argument("--unreadable-output", default="data/corpus/unreadable_pages.jsonl",
                        help="Write dropped unreadable pages as JSONL audit output.")
    parser.add_argument("--page-audit-output", default="data/corpus/page_extraction_audit.jsonl",
                        help="Write per-page method/score audit output.")
    parser.add_argument("--grades", nargs="+", default=["grade_10", "grade_11", "grade_12"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unreadable_output = Path(args.unreadable_output) if args.unreadable_output else None
    page_audit_output = Path(args.page_audit_output) if args.page_audit_output else None
    if unreadable_output is not None:
        unreadable_output.parent.mkdir(parents=True, exist_ok=True)
    if page_audit_output is not None:
        page_audit_output.parent.mkdir(parents=True, exist_ok=True)

    total_docs, total_chunks, skipped = 0, 0, 0
    unreadable_pages: list[dict] = []
    audit_pages: list[dict] = []
    ocr_langs = normalize_ocr_langs(args.ocr_langs)

    print("Extractor order : poppler_bbox -> poppler_plain -> pymupdf_words -> pdfplumber_plain -> ocr")
    print(f"OCR languages   : {ocr_langs}")
    print(f"Poppler present : {command_exists('pdftotext')}")
    print(f"PyMuPDF present : {fitz is not None}")
    print(f"pdfplumber      : {pdfplumber is not None}")

    with output_path.open("w", encoding="utf-8") as out_file:
        for grade in args.grades:
            grade_dir = input_dir / grade
            if not grade_dir.exists():
                print(f"⚠️  Skipping missing folder: {grade_dir}")
                continue

            pdfs = sorted(grade_dir.glob("*.pdf"))
            print(f"\n{grade}: {len(pdfs)} PDFs found")

            for pdf_path in pdfs:
                print(f"  Processing: {pdf_path.name}")
                chunks, page_stats = process_pdf(
                    pdf_path,
                    args.corpus_version,
                    min_page_quality=args.min_page_quality,
                    min_paragraph_len=args.min_paragraph_len,
                    ocr_fallback=args.ocr_fallback,
                    ocr_primary=args.ocr_primary,
                    ocr_langs=ocr_langs,
                    ocr_dpi=args.ocr_dpi,
                    skip_front_matter=args.skip_front_matter,
                    max_front_matter_pages=args.max_front_matter_pages,
                    unreadable_pages=unreadable_pages,
                    audit_pages=audit_pages,
                )

                avg_chunk_len = (sum(len(item["text"]) for item in chunks) / len(chunks)) if chunks else 0.0

                if not chunks:
                    print("  No valid chunks extracted")
                    print(
                        "       pages_total={pages_total} pages_kept={pages_kept} "
                        "pages_dropped_quality={pages_dropped_quality} pages_ocr_used={pages_ocr_used} "
                        "pages_skipped_front_matter={pages_skipped_front_matter} "
                        "avg_chunk_len={avg_chunk_len:.1f}".format(
                            avg_chunk_len=avg_chunk_len,
                            **page_stats,
                        )
                    )
                    skipped += 1
                    continue

                for record in chunks:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"       {len(chunks)} chunks extracted")
                print(
                    "       pages_total={pages_total} pages_kept={pages_kept} "
                    "pages_dropped_quality={pages_dropped_quality} pages_ocr_used={pages_ocr_used} "
                    "pages_skipped_front_matter={pages_skipped_front_matter} "
                    "avg_chunk_len={avg_chunk_len:.1f}".format(
                        avg_chunk_len=avg_chunk_len,
                        **page_stats,
                    )
                )
                total_docs += 1
                total_chunks += len(chunks)

    if unreadable_output is not None:
        with unreadable_output.open("w", encoding="utf-8") as out_unreadable:
            for item in unreadable_pages:
                out_unreadable.write(json.dumps(item, ensure_ascii=False) + "\n")

    if page_audit_output is not None:
        with page_audit_output.open("w", encoding="utf-8") as out_audit:
            for item in audit_pages:
                out_audit.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 48}")
    print(f"PDFs processed : {total_docs}")
    print(f"Scanned/skipped: {skipped}")
    print(f"Total chunks   : {total_chunks}")
    print(f"Output file    : {output_path}")
    if unreadable_output is not None:
        print(f"Unreadable log : {unreadable_output} ({len(unreadable_pages)} pages)")
    if page_audit_output is not None:
        print(f"Page audit log : {page_audit_output} ({len(audit_pages)} pages)")
    print(f"{'=' * 48}")
    print("\nNext step:")
    print("  python scripts/build_index.py \\")
    print(f"      --input {output_path} \\")
    print("      --output-dir data/kankor_index \\")
    print("      --embedding-backend e5")


if __name__ == "__main__":
    main()