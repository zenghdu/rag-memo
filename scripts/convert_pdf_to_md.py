#!/usr/bin/env python3
"""Convert a PDF to Markdown using both `pdf_to_md` and `ocr` utilities.

Creates two output files next to the PDF:
- <name>.md            (from `pdf_to_md.pdf_to_markdown`)
- <name>_ocr.md        (OCR-based extraction via `ocr.ocr_from_bytes`)

Usage:
    python scripts/convert_pdf_to_md.py [path/to/file.pdf]
"""
from pathlib import Path
import sys
import io
import os

# Ensure project root is on sys.path so `from app.utils import ...` works
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import pdf_to_md, ocr


def convert_with_pdf_to_md(pdf_path: Path, out_path: Path) -> None:
    text = pdf_to_md.pdf_to_markdown(pdf_path)
    out_path.write_text(text, encoding="utf-8")


def convert_with_ocr(pdf_path: Path, out_path: Path) -> bool:
    pages_text = []

    # Try PyMuPDF (fitz) first
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            page_text = ocr.ocr_from_bytes(img_bytes)
            pages_text.append(page_text)
        doc.close()
    except Exception:
        # Fallback: pdf2image
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(str(pdf_path), dpi=200)
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                page_text = ocr.ocr_from_bytes(buf.read())
                pages_text.append(page_text)
        except Exception:
            return False

    full_text = "\n\n".join(p for p in pages_text if p)
    out_path.write_text(full_text, encoding="utf-8")
    return True


def main(argv):
    pdf_path = Path(argv[1]) if len(argv) > 1 else Path("paper/FAISS_GPU_2017.pdf")
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    out_dir = pdf_path.parent
    base = pdf_path.stem

    pdf_out = out_dir / f"{base}.md"
    ocr_out = out_dir / f"{base}_ocr.md"

    print("Converting with pdf_to_md...")
    try:
        convert_with_pdf_to_md(pdf_path, pdf_out)
        print(f"Saved: {pdf_out}")
    except Exception as e:
        print(f"pdf_to_md conversion failed: {e}", file=sys.stderr)

    print("Converting with OCR (fitz -> pdf2image fallback)...")
    ok = convert_with_ocr(pdf_path, ocr_out)
    if ok:
        print(f"Saved: {ocr_out}")
    else:
        print("OCR conversion skipped (no renderer available).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
