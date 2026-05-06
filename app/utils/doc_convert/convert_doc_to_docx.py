from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional


def convert_doc_to_docx(
    doc_path: str | Path,
    out_path: Optional[str | Path] = None,
    use_libreoffice: bool = False,
) -> Path:
    """将 `.doc` 文件转换为 `.docx`。

    当前实现固定使用 Windows 上的 Word COM（pywin32）。

    Args:
        doc_path: 输入 `.doc` 文件路径。
        out_path: 可选，输出 `.docx` 路径；不传则使用相同目录同名 `.docx`。
        use_libreoffice: 保留参数，为兼容旧调用方，目前不启用。

    Returns:
        输出的 `.docx` 路径（绝对）。

    Raises:
        FileNotFoundError: 输入文件不存在。
        RuntimeError: 当前平台不支持，或 pywin32 / Word COM 转换失败。
    """
    doc_path = Path(doc_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {doc_path}")

    if doc_path.suffix.lower() != ".doc":
        raise ValueError("输入文件不是 .doc 格式")

    out_path = Path(out_path) if out_path is not None else doc_path.with_suffix(".docx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not sys.platform.startswith("win"):
        raise RuntimeError("convert_doc_to_docx 仅支持 Windows + pywin32 + Microsoft Word")

    try:
        import win32com.client as win32

        wdFormatXMLDocument = 12  # Word 保存为 .docx 的格式常量

        word = win32.gencache.EnsureDispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(doc_path))
        doc.SaveAs(str(out_path), FileFormat=wdFormatXMLDocument)
        doc.Close(False)
        word.Quit()
        return out_path.resolve()
    except Exception as e:  # pragma: no cover - Windows COM may not be available in CI
        raise RuntimeError(f"PyWin32 / Word COM 转换失败: {e}") from e
