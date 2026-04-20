import re
from pathlib import Path

import pdfplumber
import fitz  # PyMuPDF


_SECTION_HEADERS = re.compile(
    r"^(experience|education|skills|summary|objective|projects|certifications|awards)",
    re.IGNORECASE | re.MULTILINE,
)


def _confidence(text: str) -> float:
    """Heuristic confidence: fraction of words longer than 2 chars + section header presence."""
    words = text.split()
    if not words:
        return 0.0
    long_words = sum(1 for w in words if len(w) > 2)
    word_score = long_words / len(words)
    header_score = min(1.0, len(_SECTION_HEADERS.findall(text)) / 3)
    return 0.7 * word_score + 0.3 * header_score


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text using pdfplumber; fall back to PyMuPDF when confidence < 0.65."""
    path = Path(file_path)
    text = ""

    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages)
    except Exception:
        text = ""

    if _confidence(text) >= 0.65:
        return text.strip()

    # PyMuPDF fallback (better on scanned/structured PDFs)
    try:
        doc = fitz.open(str(path))
        fallback = "\n".join(page.get_text() for page in doc)
        doc.close()
        return fallback.strip()
    except Exception:
        return text.strip()
