from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def docx_to_md(src: str, dst: Optional[str] = None, fmt: str = "gfm") -> str:
    """
    使用系统安装的 `pandoc` 将 `.docx` 转换为 Markdown 文件。

    参数:
    - src: 源 docx 文件路径。
    - dst: 可选，输出 md 文件路径。若为 None，则在源文件同目录生成同名 `.md` 文件。
    - fmt: 输出 markdown 格式，默认使用 `gfm`（GitHub Flavored Markdown）。

    返回:
    - 生成的 md 文件的绝对路径（字符串）。

    注意:
    - 该函数依赖系统中可用的 `pandoc` 可执行文件。若未安装会抛出 RuntimeError。
    - 暂不实现图片自动提取（`--extract-media`）。
    """

    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"source file not found: {src}")

    if dst is None:
        dst_path = src_path.with_suffix(".md")
    else:
        dst_path = Path(dst)

    pandoc_exe = shutil.which("pandoc")
    if not pandoc_exe:
        raise RuntimeError(
            "pandoc not found in PATH. Please install pandoc: https://pandoc.org/installing.html"
        )

    cmd = [pandoc_exe, str(src_path), "-t", fmt, "-o", str(dst_path)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"pandoc conversion failed: {stderr}") from e

    return str(dst_path.resolve())
