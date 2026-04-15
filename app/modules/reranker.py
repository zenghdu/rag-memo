import requests
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.utils.logger import logger

class Reranker:
    """重排序模块 (Module: Reranker)"""
    
    def __init__(self, model_name: str = None, top_n: int = 5):
        self.model_name = model_name or settings.reranker_model
        self.top_n = top_n
        self.api_key = settings.reranker_api_key
        self.base_url = settings.reranker_base_url

    def run(self, query: str, scored_docs: List[Tuple[LCDocument, float]]) -> List[Tuple[LCDocument, float]]:
        """执行重排序流程 (Cross-Encoder 模型)"""
        if not scored_docs:
            return []

        logger.debug(f"Reranking for query: {query}")
        
        # 准备 Reranker 输入
        # 这里使用标准 OpenAI 兼容的 Rerank API (如 SiliconFlow/OpenRouter 支持的接口)
        # 或者模拟一个标准的 rerank 请求
        try:
            reranked_docs = self._call_rerank_api(query, scored_docs)
            # 返回重排后的 top_n 个结果
            return reranked_docs[:self.top_n]
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            # Rerank 失败时回退到原始召回结果的前 top_n
            return scored_docs[:self.top_n]

    def _call_rerank_api(self, query: str, scored_docs: List[Tuple[LCDocument, float]]) -> List[Tuple[LCDocument, float]]:
        """调用远程 Reranker API"""
        # 提取文本内容
        docs_text = [doc.page_content for doc, _ in scored_docs]
        
        # 这里的接口逻辑根据实际 Qwen3-Reranker-8B 提供的 API 规范调整
        # 通常是 /rerank 或兼容接口
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": docs_text,
            "top_n": self.top_n,
            "return_documents": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 尝试通过 /rerank 接口 (OpenAPI 规范)
        url = f"{self.base_url.rstrip('/')}/rerank"
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Rerank API returned status {response.status_code}")
            return scored_docs

        data = response.json()
        # 解析返回的分数和索引
        # 示例: [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}]
        results = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            doc = scored_docs[idx][0]
            doc.metadata["rerank_score"] = score
            results.append((doc, score))
            
        return results
