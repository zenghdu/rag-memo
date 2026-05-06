"""批量文档摄入脚本 — 读取指定文件夹下所有文件，完成 文档导入 → 切片 → 向量化 全流程"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

from app.core.config import settings
from app.db.mysql import init_db
from app.services.pipeline import PipelineService
from app.services.loader import SUPPORTED_EXTS
from app.utils.logger import logger


class BatchIngestRunner:
    """批量文档摄入执行器"""

    def __init__(self):
        self.pipeline = PipelineService()
        self.supported_exts = SUPPORTED_EXTS

    def collect_files(self, folder: Path, recursive: bool = True) -> List[Path]:
        """收集指定文件夹下所有支持的文件"""
        files: List[Path] = []
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"路径不是文件夹: {folder}")

        pattern = "**/*" if recursive else "*"
        for file_path in folder.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_exts:
                files.append(file_path)

        files.sort(key=lambda p: p.name)
        return files

    def run(self, folder: Path, recursive: bool = True) -> dict:
        """执行批量摄入"""
        files = self.collect_files(folder, recursive=recursive)

        if not files:
            logger.warning(f"在 {folder} 中未找到支持的文件")
            logger.info(f"支持的文件类型: {', '.join(sorted(self.supported_exts))}")
            return {"total": 0, "succeeded": 0, "failed": 0, "results": []}

        logger.info(f"共发现 {len(files)} 个待处理文件")
        logger.info("=" * 60)

        results = []
        succeeded = 0
        failed = 0
        total_start = time.time()

        for idx, file_path in enumerate(files, 1):
            logger.info(f"[{idx}/{len(files)}] 正在处理: {file_path.name}")
            try:
                t0 = time.time()
                result = self.pipeline.ingest(file_path)
                elapsed = time.time() - t0

                result["elapsed_seconds"] = round(elapsed, 2)
                result["status"] = "success"
                results.append(result)
                succeeded += 1

                logger.info(
                    f"  ✓ 完成 | 页数: {result.get('pages', '?')} | "
                    f"切片数: {result.get('chunks', '?')} | "
                    f"耗时: {elapsed:.2f}s"
                )
            except Exception as exc:
                failed += 1
                err_info = {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "status": "failed",
                    "error": str(exc),
                }
                results.append(err_info)
                logger.error(f"  ✗ 失败 | 错误: {exc}")

        total_elapsed = time.time() - total_start

        summary = {
            "total": len(files),
            "succeeded": succeeded,
            "failed": failed,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": results,
        }

        logger.info("=" * 60)
        logger.info(
            f"批量摄入完成 | 共 {summary['total']} 个文件 | "
            f"成功 {summary['succeeded']} | "
            f"失败 {summary['failed']} | "
            f"总耗时 {summary['total_elapsed_seconds']:.2f}s"
        )

        return summary


def main():
    """入口函数 — 从命令行参数获取文件夹路径"""
    if len(sys.argv) < 2:
        print(f"用法: python -m app.batch_ingest <文件夹路径> [--no-recursive]")
        print(f"示例: python -m app.batch_ingest ./paper")
        print(f"      python -m app.batch_ingest ./paper --no-recursive")
        print(f"")
        print(f"支持的文件类型: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    recursive = "--no-recursive" not in sys.argv

    print(f"目标文件夹: {folder}")
    print(f"递归扫描: {'是' if recursive else '否'}")
    print(f"Milvus 集合: {settings.milvus_collection}")
    print()

    init_db()

    runner = BatchIngestRunner()
    runner.run(folder, recursive=recursive)


if __name__ == "__main__":
    main()
