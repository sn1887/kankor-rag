from __future__ import annotations

import html
import re
import unicodedata

CHAR_NORMALIZATION_MAP = str.maketrans(
    {
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
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)

BIDI_CONTROL_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_./+-]{0,12}\b")
SHORT_LATIN_JUNK_RE = re.compile(r"\b[A-Za-z]{1,5}\b")
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

LATIN_KEEP_TOKENS = {
    "dna",
    "rna",
    "atp",
    "adp",
    "ph",
    "cm",
    "mm",
    "km",
    "kg",
    "mg",
    "ml",
    "o2",
    "co2",
    "h2o",
    "na",
    "cl",
    "ca",
    "fe",
    "cu",
    "tv",
    "it",
    "pc",
}


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
