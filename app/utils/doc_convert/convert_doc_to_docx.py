from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Optional
import tempfile
import shutil


def convert_doc_to_docx(
    doc_path: str | Path,
    out_path: Optional[str | Path] = None,
    use_libreoffice: bool = False,
) -> Path:
    """将 `.doc` 文件转换为 `.docx`。

    优先在 Windows 上使用 Word COM（pywin32），失败或在非 Windows 环境下则回退到 LibreOffice 的 `soffice`。

    Args:
        doc_path: 输入 `.doc` 文件路径。
        out_path: 可选，输出 `.docx` 路径；不传则使用相同目录同名 `.docx`。
        use_libreoffice: 强制使用 LibreOffice（即使在 Windows 上也跳过 COM）。

    Returns:
        输出的 `.docx` 路径（绝对）。

    Raises:
        FileNotFoundError: 输入文件不存在。
        RuntimeError: 当所有转换方式都失败时抛出。
    """
    doc_path = Path(doc_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {doc_path}")

    if doc_path.suffix.lower() != ".doc":
        raise ValueError("输入文件不是 .doc 格式")

    out_path = Path(out_path) if out_path is not None else doc_path.with_suffix(".docx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None

    if sys.platform.startswith("win") and not use_libreoffice:
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
            last_exc = e

    # 使用仓库内的 doc_to_md 二进制作为回退：生成中间 HTML，再用 pandoc 转为 .docx
    bin_dir = Path(__file__).resolve().parent
    doc_to_md_bin = bin_dir / ("doc_to_md.exe" if sys.platform.startswith("win") else "doc_to_md")

    if not doc_to_md_bin.exists():
        if last_exc is not None:
            raise RuntimeError(f"PyWin32 转换失败: {last_exc}; 本地转换工具未找到: {doc_to_md_bin}")
        raise FileNotFoundError(f"本地转换工具未找到: {doc_to_md_bin}")

    temp_md = None
    temp_html = None
    try:
        # 创建临时文件以接收中间结果
        tmp_md = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
        tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp_md.close()
        tmp_html.close()
        temp_md = Path(tmp_md.name)
        temp_html = Path(tmp_html.name)

        cmd = [str(doc_to_md_bin), str(doc_path), str(temp_md), str(temp_html)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"本地 doc_to_md 转换失败 (exit={res.returncode}):\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )

        # 使用 pandoc 将 HTML 转为 docx
        pandoc_cmd = [
            "pandoc",
            "-f",
            "html",
            "-t",
            "docx",
            "-o",
            str(out_path),
            str(temp_html),
        ]

        pandoc_res = subprocess.run(pandoc_cmd, capture_output=True, text=True)
        if pandoc_res.returncode != 0:
            raise RuntimeError(
                f"pandoc 转换失败 (exit={pandoc_res.returncode}):\nstdout:\n{pandoc_res.stdout}\nstderr:\n{pandoc_res.stderr}"
            )

        if not out_path.exists():
            raise RuntimeError("pandoc 执行成功但未找到输出文件")

        return out_path.resolve()
    finally:
        # 清理临时文件
        for p in (temp_md, temp_html):
            try:
                if p is not None and p.exists():
                    p.unlink()
            except Exception:
                pass
