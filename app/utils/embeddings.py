"""Embedding 模型 — 使用 OpenAI 兼容接口调用 Qwen3 Embedding"""

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """返回配置好的 Embedding 模型实例。"""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.embedding_api_key,
        openai_api_base=settings.embedding_base_url,
        check_embedding_ctx_length=False,   # 非 OpenAI 模型不支持 token 分片
    )
