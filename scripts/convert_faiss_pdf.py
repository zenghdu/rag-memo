#!/usr/bin/env python3
"""
Convert a PDF to Markdown using two methods:
- text extraction via `app.utils.pdf_to_md.pdf_to_markdown`
- OCR via `app.utils.ocr.ocr_from_bytes` (requires `pdf2image` or `PyMuPDF` + `Pillow`)

Creates two files next to the source PDF:
- <basename>_pdftotext.md
- <basename>_ocr.md
"""
import io
import sys
from pathlib import Path

DEFAULT_PDF = Path("paper") / "FAISS_GPU_2017.pdf"


def convert_with_pdf_to_md(pdf_path: Path) -> str:
    try:
        from app.utils.pdf_to_md import pdf_to_markdown
    except Exception as e:
        raise RuntimeError("failed to import app.utils.pdf_to_md") from e

    return pdf_to_markdown(pdf_path)


def convert_with_ocr(pdf_path: Path, dpi: int = 200) -> str:
    try:
        from app.utils.ocr import ocr_from_bytes
    except Exception as e:
        raise RuntimeError("failed to import app.utils.ocr") from e

    page_texts = []

    # try pdf2image first
    try:
        from pdf2image import convert_from_path

        pil_images = convert_from_path(str(pdf_path), dpi=dpi)
        for page_num, pil_img in enumerate(pil_images, start=1):
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            text = ocr_from_bytes(img_bytes)
            page_texts.append(f"<!-- PAGE {page_num} -->\n{text}")
        return "\n\n".join(page_texts)
    except Exception:
        pass

    # fallback: try PyMuPDF
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)
            try:
                img_bytes = pix.tobytes("png")
            except TypeError:
                img_bytes = pix.tobytes()
            text = ocr_from_bytes(img_bytes)
            page_texts.append(f"<!-- PAGE {page_index+1} -->\n{text}")
        return "\n\n".join(page_texts)
    except Exception as e:
        raise RuntimeError(
            "No raster backend available for OCR (need pdf2image or PyMuPDF)."
        ) from e


def write_md(content: str, output_path: Path):
    output_path.write_text(content or "", encoding="utf-8")


def main(argv=None):
    argv = argv or sys.argv[1:]
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using text extraction and OCR"
    )
    parser.add_argument("pdf", nargs="?", default=str(DEFAULT_PDF), help="Path to the PDF file")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rasterization when using OCR")
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)

    base_name = pdf_path.stem
    out_dir = pdf_path.parent

    # method 1: text extraction
    try:
        text_output = convert_with_pdf_to_md(pdf_path)
    except Exception as e:
        text_output = f"ERROR: {e}"
    out_text_file = out_dir / f"{base_name}_pdftotext.md"
    write_md(text_output, out_text_file)
    print(f"Wrote {out_text_file}")

    # method 2: OCR
    try:
        ocr_output = convert_with_ocr(pdf_path, dpi=args.dpi)
    except Exception as e:
        ocr_output = f"ERROR: {e}"
    out_ocr_file = out_dir / f"{base_name}_ocr.md"
    write_md(ocr_output, out_ocr_file)
    print(f"Wrote {out_ocr_file}")


if __name__ == "__main__":
    main()
