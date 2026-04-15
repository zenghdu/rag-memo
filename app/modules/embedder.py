from typing import List, Dict, Any, Optional
import time
from langchain_core.documents import Document as LCDocument
from langchain_milvus import Milvus

import app.db.milvus  # noqa: F401  兼容性补丁
from app.core.config import settings
from app.services.embeddings import get_embeddings
from app.utils.logger import logger

class Embedder:
    """向量化与入库模块 (Module: Embedder)"""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.milvus_collection
        self.embeddings = get_embeddings()
        
        # HNSW 索引参数 (Milvus)
        self.index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }

    def run(self, chunks: List[LCDocument]) -> Dict[str, Any]:
        """执行向量化与入库"""
        if not chunks:
            return {"count": 0, "status": "skipped"}

        # 初始化 Milvus 存储
        vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": settings.milvus_uri, "token": settings.milvus_token},
            index_params=self.index_params,
            auto_id=True,
            drop_old=True # 测试环境下为了确保 schema 干净，可以考虑 drop_old 或在外部处理
        )

        # 写入文档
        ids = vector_store.add_documents(chunks)
        
        return {
            "count": len(ids),
            "ids": ids[:5], # 仅记录前5个ID
            "collection": self.collection_name
        }
