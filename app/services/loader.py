from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document as LCDocument

from app.utils.ocr import ocr_from_bytes
from app.utils.logger import logger

# 支持的文件类型
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_DOC_EXTS = {".docx", ".doc"} # 预留
SUPPORTED_TEXT_EXTS = {".txt", ".md"} # 预留
SUPPORTED_EXTS = SUPPORTED_PDF_EXTS | SUPPORTED_IMAGE_EXTS | SUPPORTED_DOC_EXTS | SUPPORTED_TEXT_EXTS

class Loader:
    """文档加载模块 (Module: Loader)"""
    
    def __init__(self, min_text_length: int = 30):
        self.min_text_length = min_text_length

    def run(self, file_path: Path) -> List[LCDocument]:
        """执行加载流程"""
        ext = file_path.suffix.lower()
        logger.debug(f"Loading document: {file_path} (ext: {ext})")
        
        if ext in SUPPORTED_PDF_EXTS:
            return self._load_pdf(file_path)
        elif ext in SUPPORTED_IMAGE_EXTS:
            return self._load_image(file_path)
        elif ext in SUPPORTED_TEXT_EXTS:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, pdf_path: Path) -> List[LCDocument]:
        documents: List[LCDocument] = []
        with fitz.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = (page.get_text("text") or "").strip()
                parser = "pymupdf"

                # OCR 触发逻辑
                if len(text) < self.min_text_length:
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        text = ocr_text
                        parser = "pymupdf+ocr"

                if not text:
                    continue

                documents.append(
                    LCDocument(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "page": page_index,
                            "parser": parser,
                            "file_type": "pdf",
                        },
                    )
                )
        return documents

    def _ocr_page(self, page: fitz.Page, dpi: int = 300) -> str:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")
        return ocr_from_bytes(image_bytes)

    def _load_image(self, image_path: Path) -> List[LCDocument]:
        image_bytes = image_path.read_bytes()
        text = ocr_from_bytes(image_bytes)
        if not text.strip():
            return []

        return [
            LCDocument(
                page_content=text,
                metadata={
                    "source": str(image_path),
                    "filename": image_path.name,
                    "parser": "rapidocr",
                    "file_type": "image",
                },
            )
        ]

    def _load_text(self, path: Path) -> List[LCDocument]:
        content = path.read_text(encoding="utf-8")
        return [
            LCDocument(
                page_content=content,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": "text",
                }
            )
        ]
