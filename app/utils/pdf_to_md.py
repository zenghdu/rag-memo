from pathlib import Path
from typing import Union


def extract_text_from_pdf(path: Union[str, Path]) -> str:
    """Extract text from a PDF using available backends.

    Tries to use pdfminer.six first, then falls back to PyPDF2.
    """
    path = str(path)
    try:
        from pdfminer.high_level import extract_text

        return extract_text(path)
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    except Exception:
        raise RuntimeError(
            "No PDF extraction backend available (need pdfminer.six or PyPDF2)"
        )


def pdf_to_markdown(path: Union[str, Path]) -> str:
    """Convert a PDF file to a Markdown-like plain text string.

    This function extracts text from the PDF and returns a simple markdown-friendly
    string (preserves paragraphs). It intentionally keeps conversion simple so
    it's usable without heavy external tooling.
    """
    text = extract_text_from_pdf(path)

    # Minimal cleanup: strip trailing whitespace and ensure Unix newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    return text


__all__ = ["extract_text_from_pdf", "pdf_to_markdown"]
