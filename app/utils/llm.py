"""LLM 工厂 — 提供 LLM 实例"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例。"""
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        temperature=0.2,
        streaming=True,
    )
