import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from app.core.config import settings, DATA_DIR
from app.services.pipeline import PipelineService
from app.db.mysql import init_db
from app.utils.logger import console

class TestModularRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试前初始化"""
        console.print("\n[bold magenta]🚀 Starting Modular RAG Reconstruction Tests[/bold magenta]")
        # 强制开启调试模式查看详细输出
        settings.debug_pipeline = True
        # 初始化数据库表
        init_db()
        cls.pipeline = PipelineService()
        cls.test_pdf = DATA_DIR / "Understanding_Climate_Change.pdf"

    def test_01_ingest_pipeline(self):
        """测试 1: 完整的摄入流程 (Loader -> Chunker -> Embedder)"""
        console.print("\n[bold cyan]=== Test 01: Ingest Pipeline ===[/bold cyan]")
        
        if not self.test_pdf.exists():
            self.skipTest(f"Test file {self.test_pdf} not found")

        result = self.pipeline.ingest(self.test_pdf)
        
        self.assertIn("run_id", result)
        self.assertEqual(result["filename"], self.test_pdf.name)
        self.assertGreater(result["pages"], 0)
        self.assertGreater(result["chunks"], 0)
        
        # 检查模块结果
        details = {r.module_name: r for r in result["details"]}
        self.assertEqual(details["loader"].status, "success")
        self.assertEqual(details["chunker"].status, "success")
        self.assertEqual(details["embedder"].status, "success")
        
        console.print(f"\n✅ Ingest Pipeline Test Passed: {result['chunks']} chunks inserted.")

    def test_02_chat_pipeline(self):
        """测试 2: 完整的问答流程 (Retriever -> Reranker -> Context -> LLM)"""
        console.print("\n[bold cyan]=== Test 02: Chat Pipeline ===[/bold cyan]")
        
        query = "What is the greenhouse effect?"
        answer = self.pipeline.chat(query)
        
        self.assertIsNotNone(answer)
        self.assertGreater(len(answer), 20)
        
        console.print("\n✅ Chat Pipeline Test Passed.")

    def test_03_error_handling(self):
        """测试 3: 异常处理 (不存在的文件)"""
        console.print("\n[bold cyan]=== Test 03: Error Handling ===[/bold cyan]")
        
        fake_path = Path("non_existent_file.pdf")
        with self.assertRaises(Exception):
            self.pipeline.ingest(fake_path)
            
        console.print("\n✅ Error Handling Test Passed.")

if __name__ == "__main__":
    unittest.main()
