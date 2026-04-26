"""Embedding 模型工厂。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def _normalize_openai_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


class ConcurrentOpenAIEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.embedding_api_key,
            openai_api_base=_normalize_openai_base_url(settings.embedding_base_url),
            check_embedding_ctx_length=False,
            chunk_size=max(1, int(settings.embedding_batch_size)),
        )
        self.concurrency = max(1, int(settings.embedding_concurrency))
        self.batch_size = max(1, int(settings.embedding_batch_size))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        if len(batches) == 1 or self.concurrency == 1:
            return self.client.embed_documents(texts)

        max_workers = min(self.concurrency, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_vectors = list(executor.map(self.client.embed_documents, batches))

        vectors: List[List[float]] = []
        for batch in batch_vectors:
            vectors.extend(batch)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)


def get_embeddings() -> Embeddings:
    """返回配置好的 Embedding 模型实例。"""
    return ConcurrentOpenAIEmbeddings()
