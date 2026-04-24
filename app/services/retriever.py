from typing import List, Tuple
from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.db.milvus import MilvusVectorStore
from app.utils.logger import logger


class Retriever:
    """召回模块 (Module: Retriever)"""

    def __init__(self, collection_name: str = None, top_k: int = None):
        self.collection_name = collection_name or settings.milvus_collection
        self.top_k = top_k or settings.retriever_top_k
        self.vector_store = MilvusVectorStore(collection_name=self.collection_name)
        self.last_search_info = self.vector_store.observability() | {"top_k": self.top_k, "returned": 0}

    def run(self, query: str) -> List[Tuple[LCDocument, float]]:
        """执行召回流程 (向量检索)"""
        logger.debug(f"Retrieving for query: {query}")
        results = self.vector_store.similarity_search_with_score(query=query, k=self.top_k)
        self.last_search_info = self.vector_store.observability() | {
            "top_k": self.top_k,
            "returned": len(results),
        }

        scored_docs = []
        for doc, score in results:
            doc.metadata["recall_score"] = score
            doc.metadata["recall_score_display"] = score
            doc.metadata["recall_score_raw"] = doc.metadata.get("ann_raw_score")
            doc.metadata["recall_score_kind"] = "normalized_similarity"
            scored_docs.append((doc, score))
        return scored_docs
