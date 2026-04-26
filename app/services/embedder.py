from typing import List, Dict, Any, Optional
from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.db.milvus import MilvusVectorStore


class Embedder:
    """向量化与入库模块 (Module: Embedder)"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.milvus_collection
        self.vector_store = MilvusVectorStore(collection_name=self.collection_name)

    def run(self, chunks: List[LCDocument]) -> Dict[str, Any]:
        """执行向量化与入库"""
        if not chunks:
            return {"count": 0, "status": "skipped", "ids": []}

        ids = self.vector_store.add_documents(chunks)
        return {
            "count": len(ids),
            "ids": ids[:5],
            "all_ids": ids,
            "collection": self.collection_name,
        }
