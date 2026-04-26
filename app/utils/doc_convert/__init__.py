from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

DOC_TO_MD_BIN = Path(__file__).resolve().parent / "doc_to_md"


def convert_doc_to_md(
    doc_path: str | Path,
    md_path: str | Path,
    html_path: Optional[str | Path] = None,
) -> Path:
    """调用本地 C 转换程序，将 .doc 转为 .md。

    Args:
        doc_path: 输入 .doc 文件路径
        md_path: 输出 .md 文件路径
        html_path: 可选，中间 .html 输出路径；不传则由二进制使用临时文件

    Returns:
        生成的 Markdown 文件路径
    """
    if not DOC_TO_MD_BIN.exists():
        raise FileNotFoundError(f"找不到转换程序: {DOC_TO_MD_BIN}")

    doc_path = Path(doc_path).resolve()
    md_path = Path(md_path).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(DOC_TO_MD_BIN), str(doc_path), str(md_path)]

    if html_path is not None:
        html_path = Path(html_path).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(str(html_path))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DOC 转 Markdown 转换失败: {doc_path}\n"
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not md_path.exists():
        raise RuntimeError(f"转换程序执行成功，但未找到输出文件: {md_path}")

    return md_path
