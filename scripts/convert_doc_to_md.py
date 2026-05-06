from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
src_docx = repo / "app" / "data" / "江苏监管局" / "规范性文件" / "关于提示异地贷款业务风险的通知.docx"
if not src_docx.exists():
    print(f"source docx not found: {src_docx}")
    sys.exit(2)

dst_md = repo / "app" / "data" / "test" / "关于提示异地贷款业务风险的通知_from_doc.md"

try:
    from utils.docx_to_md import docx_to_md
except Exception:
    # fallback to load file directly
    import importlib.util
    mod_path = repo / "utils" / "docx_to_md.py"
    spec = importlib.util.spec_from_file_location("utils_docx_to_md", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    docx_to_md = mod.docx_to_md

print("Converting:", src_docx)
out = docx_to_md(str(src_docx), dst=str(dst_md), remove_blockquote=True)
print("Wrote:", out)
