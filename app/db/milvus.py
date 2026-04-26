"""Milvus 连接、Collection 生命周期与向量读写"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document as LCDocument
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.core.config import settings
from app.utils.embeddings import get_embeddings

MILVUS_ALIAS = "rag_memo"
PRIMARY_FIELD = "id"
VECTOR_FIELD = "vector"
TEXT_FIELD = "text"
METADATA_FIELD = "metadata_json"


def connect(alias: str = MILVUS_ALIAS) -> str:
    """确保 Milvus 连接存在。"""
    if not connections.has_connection(alias):
        kwargs: Dict[str, Any] = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        connections.connect(alias=alias, **kwargs)
    return alias


class MilvusVectorStore:
    """使用 pymilvus 显式管理 collection schema / index / search。"""

    DISTANCE_METRICS = {"COSINE", "L2", "JACCARD", "HAMMING"}
    SIMILARITY_METRICS = {"IP"}

    def __init__(self, collection_name: Optional[str] = None, alias: str = MILVUS_ALIAS):
        self.collection_name = collection_name or settings.milvus_collection
        self.alias = connect(alias)
        self.embeddings = get_embeddings()

    @property
    def index_params(self) -> Dict[str, Any]:
        return {
            "metric_type": settings.milvus_metric_type,
            "index_type": settings.milvus_index_type,
            "params": {
                "M": settings.hnsw_m,
                "efConstruction": settings.hnsw_ef_construction,
            },
        }

    @property
    def search_params(self) -> Dict[str, Any]:
        return {
            "metric_type": settings.milvus_metric_type,
            "params": {"ef": settings.hnsw_ef_search},
        }

    def ensure_collection(self) -> Collection:
        """显式创建 collection/schema/index；已存在则直接复用。"""
        if utility.has_collection(self.collection_name, using=self.alias):
            collection = Collection(self.collection_name, using=self.alias)
            if not collection.indexes:
                collection.create_index(VECTOR_FIELD, self.index_params)
            return collection

        fields = [
            FieldSchema(name=PRIMARY_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name=TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=METADATA_FIELD, dtype=DataType.JSON),
            FieldSchema(
                name=VECTOR_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.milvus_vector_dim,
            ),
        ]
        schema = CollectionSchema(fields=fields, description="RAG Memo chunks")
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.alias,
            consistency_level="Strong",
        )
        collection.create_index(VECTOR_FIELD, self.index_params)
        return collection

    def add_documents(self, documents: List[LCDocument], *, flush: Optional[bool] = None) -> List[str]:
        if not documents:
            return []

        collection = self.ensure_collection()
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        payload = [
            [int(doc.metadata.get("document_id", 0)) for doc in documents],
            [int(doc.metadata.get("chunk_index", idx)) for idx, doc in enumerate(documents)],
            [int(doc.metadata.get("page", -1) or -1) for doc in documents],
            [str(doc.metadata.get("filename", ""))[:512] for doc in documents],
            [str(doc.metadata.get("source", ""))[:2048] for doc in documents],
            [text[:65535] for text in texts],
            [self._normalize_metadata(doc.metadata) for doc in documents],
            vectors,
        ]

        result = collection.insert(payload)
        if flush if flush is not None else settings.milvus_auto_flush:
            collection.flush()
        return [str(pk) for pk in result.primary_keys]

    def delete_by_document_id(self, document_id: int, *, flush: bool = True) -> None:
        collection = self.ensure_collection()
        collection.load()
        collection.delete(expr=f"document_id == {int(document_id)}")
        if flush:
            collection.flush()

    def flush(self) -> None:
        collection = self.ensure_collection()
        collection.flush()

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[LCDocument, float]]:
        collection = self.ensure_collection()
        collection.load()
        query_vector = self.embeddings.embed_query(query)
        expr = self._build_expr(filters)
        search_kwargs: Dict[str, Any] = {
            "data": [query_vector],
            "anns_field": VECTOR_FIELD,
            "param": self.search_params,
            "limit": k,
            "output_fields": [
                "document_id",
                "chunk_index",
                "page_num",
                "filename",
                "source",
                TEXT_FIELD,
                METADATA_FIELD,
            ],
        }
        if expr:
            search_kwargs["expr"] = expr
        search_result = collection.search(**search_kwargs)

        hits = search_result[0] if search_result else []
        docs_with_scores: List[Tuple[LCDocument, float]] = []
        for hit in hits:
            entity = hit.entity
            raw_score = float(hit.distance)
            normalized_score = self._normalize_score(raw_score)
            metadata = self._decode_metadata(entity.get(METADATA_FIELD))
            metadata.update(
                {
                    "document_id": entity.get("document_id"),
                    "chunk_index": entity.get("chunk_index"),
                    "page": entity.get("page_num"),
                    "filename": entity.get("filename"),
                    "source": entity.get("source"),
                    "milvus_id": str(hit.id),
                    "ann_metric_type": settings.milvus_metric_type,
                    "ann_raw_score": raw_score,
                    "ann_score_kind": self.score_kind,
                    "ann_score_direction": self.score_direction,
                    "ann_normalized_score": normalized_score,
                }
            )
            doc = LCDocument(page_content=entity.get(TEXT_FIELD), metadata=metadata)
            docs_with_scores.append((doc, normalized_score))
        return docs_with_scores

    @property
    def score_kind(self) -> str:
        metric = settings.milvus_metric_type.upper()
        if metric in self.SIMILARITY_METRICS:
            return "similarity"
        return "distance"

    @property
    def score_direction(self) -> str:
        return "higher_is_better" if self.score_kind == "similarity" else "lower_is_better"

    def observability(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "collection": self.collection_name,
            "metric_type": settings.milvus_metric_type,
            "index_type": settings.milvus_index_type,
            "search_params": self.search_params,
            "score_kind": self.score_kind,
            "score_direction": self.score_direction,
            "normalized_score_desc": "higher_is_better",
            "filters": filters or {},
            "expr": self._build_expr(filters),
        }

    def _build_expr(self, filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not filters:
            return None

        clauses: List[str] = []
        document_ids = filters.get("document_ids") or []
        if document_ids:
            ids = [str(int(doc_id)) for doc_id in document_ids]
            clauses.append(f"document_id in [{', '.join(ids)}]")

        filename = filters.get("filename")
        if filename:
            clauses.append(f"filename == {json.dumps(str(filename))}")

        source = filters.get("source")
        if source:
            clauses.append(f"source == {json.dumps(str(source))}")

        return " and ".join(clauses) if clauses else None

    def _normalize_score(self, raw_score: float) -> float:
        metric = settings.milvus_metric_type.upper()
        if metric == "COSINE":
            return max(-1.0, min(1.0, 1.0 - raw_score))
        if metric == "IP":
            return raw_score
        if metric == "L2":
            return 1.0 / (1.0 + raw_score)
        return raw_score if self.score_kind == "similarity" else -raw_score

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

    def _decode_metadata(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}
