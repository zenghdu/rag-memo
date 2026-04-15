from typing import List, Dict, Any, Optional, Tuple
from langchain_milvus import Milvus
from langchain_core.documents import Document as LCDocument

import app.db.milvus  # noqa: F401  兼容性补丁
from app.core.config import settings
from app.services.embeddings import get_embeddings
from app.utils.logger import logger

class Retriever:
    """召回模块 (Module: Retriever)"""
    
    def __init__(self, collection_name: str = None, top_k: int = None):
        self.collection_name = collection_name or settings.milvus_collection
        self.top_k = top_k or settings.retriever_top_k
        self.embeddings = get_embeddings()
        
        # HNSW 搜索参数
        self.search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128},
        }

    def run(self, query: str) -> List[Tuple[LCDocument, float]]:
        """执行召回流程 (向量检索)"""
        logger.debug(f"Retrieving for query: {query}")
        
        # 初始化 Milvus 存储
        vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": settings.milvus_uri, "token": settings.milvus_token},
            search_params=self.search_params,
        )

        # 相似度检索
        results = vector_store.similarity_search_with_score(
            query=query, 
            k=self.top_k
        )
        
        # 将结果注入元数据 score
        scored_docs = []
        for doc, score in results:
            doc.metadata["recall_score"] = score
            scored_docs.append((doc, score))
            
        return scored_docs
