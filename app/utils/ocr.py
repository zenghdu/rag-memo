"""OCR 工具 — 基于 RapidOCR 实现图片文字识别"""

from __future__ import annotations

from rapidocr_onnxruntime import RapidOCR

_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def ocr_from_bytes(image_bytes: bytes) -> str:
    """
    对图片字节流执行 OCR，返回识别出的文本。
    支持 PNG / JPG / BMP / TIFF 等常见格式。
    """
    engine = _get_engine()
    result, _ = engine(image_bytes)
    if not result:
        return ""
    # result 每项: [bbox, text, confidence]
    return "\n".join(item[1] for item in result)
