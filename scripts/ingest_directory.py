from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.mysql import init_db
from app.services.loader import SUPPORTED_EXTS
from app.services.pipeline import PipelineService


def main() -> int:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "app" / "data" / "extracted"
    target_dir = target_dir.resolve()
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return 1

    files = sorted(
        path for path in target_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTS
        and not path.name.startswith('.~')
    )
    print(f"发现 {len(files)} 个可摄入文件: {target_dir}", flush=True)
    if not files:
        return 0

    init_db()
    pipeline = PipelineService()

    success = 0
    failed = 0
    for idx, path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] ingest -> {path}", flush=True)
        try:
            result = pipeline.ingest(path, flush_vector_store=False)
            print(f"  ok: pages={result['pages']} chunks={result['chunks']}", flush=True)
            success += 1
        except Exception as exc:
            print(f"  fail: {exc}", flush=True)
            failed += 1

    print("\n正在执行最终向量 flush ...", flush=True)
    pipeline.embedder.vector_store.flush()
    print(f"完成: success={success}, failed={failed}, total={len(files)}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
