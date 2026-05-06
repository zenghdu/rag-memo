from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_repo_root_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def convert_docx_with_pandoc(src: Path) -> Path:
    ensure_repo_root_on_path()
    try:
        from utils.docx_to_md import docx_to_md
        out = docx_to_md(str(src), remove_blockquote=True)
        return Path(out)
    except Exception:
        # Fallback: load module directly from file path
        import importlib.util

        repo_root = Path(__file__).resolve().parents[1]
        mod_path = repo_root / "utils" / "docx_to_md.py"
        if not mod_path.exists():
            raise
        spec = importlib.util.spec_from_file_location("utils_docx_to_md", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        out = mod.docx_to_md(str(src), remove_blockquote=True)
        return Path(out)


def convert_doc_with_doc_convert(exe_dir: Path, src: Path, dst_dir: Path, exe_path: Path | None = None) -> Path:
    # prefer provided executable path (e.g. the compiled doc_to_md.exe)
    exe = None
    if exe_path is not None:
        if exe_path.exists() and exe_path.is_file():
            exe = exe_path
        else:
            raise FileNotFoundError(f"doc_to_md executable not found at {exe_path}")
    else:
        # fallback: find executable in exe_dir
        candidates = [exe_dir / "doc_to_md.exe", exe_dir / "doc_to_md", exe_dir / "doc_to_md.py"]
        for c in candidates:
            if c.exists() and c.is_file():
                exe = c
                break

    if exe is None:
        raise FileNotFoundError(f"doc_to_md executable not found in {exe_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    out_md = dst_dir / (src.stem + ".md")

    cmd = [str(exe), str(src), str(out_md)]

    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"doc_to_md failed (code {result.returncode}): {result.stderr.decode('utf-8', errors='ignore')}"
        )

    return out_md.resolve()


def main():
    repo = Path(__file__).resolve().parents[1]

    # 1) Convert DOCX in app/data/test using utils.docx_to_md
    src_docx = repo / "app" / "data" / "test" / "关于提示异地贷款业务风险的通知.docx"
    if not src_docx.exists():
        print(f"source docx not found: {src_docx}")
    else:
        print(f"Converting DOCX -> MD: {src_docx}")
        out_md = convert_docx_with_pandoc(src_docx)
        print(f"DOCX converted to: {out_md}")

    # 2) Convert DOC (old binary) using app/utils/doc_convert/doc_to_md
    src_doc = repo / "app" / "data" / "江苏监管局" / "规范性文件" / "关于提示异地贷款业务风险的通知.doc"
    exe_dir = repo / "app" / "utils" / "doc_convert"
    dst_dir = repo / "app" / "data" / "test"

    if not src_doc.exists():
        print(f"source .doc not found: {src_doc}")
    else:
        print(f"Converting DOC -> MD using doc_convert: {src_doc}")
        # prefer the freshly compiled executable
        compiled_exe = exe_dir / "doc_to_md.exe"
        out_md2 = convert_doc_with_doc_convert(exe_dir, src_doc, dst_dir, exe_path=compiled_exe)
        print(f"DOC converted to: {out_md2}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
