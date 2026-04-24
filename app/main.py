"""FastAPI 入口 — RAG 系统 API"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.core.config import UPLOAD_DIR, settings
from app.db.mysql import init_db
from app.services.pipeline import PipelineService
from app.services.loader import SUPPORTED_EXTS

# ═══════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════

app = FastAPI(
    title="RAG-Memo Modular",
    version="2.0.0 (Modular)",
    description=(
        "基于模块化设计的 RAG 系统\n\n"
        "- 支持多模态 Loader (PDF/Image/Text)\n"
        "- 可插拔 Chunker 策略\n"
        "- 全流程状态监控 (MySQL + Terminal Console)\n"
    ),
)

# 初始化数据库
init_db()

pipeline = PipelineService()

@app.get("/")
async def root():
    return RedirectResponse("/docs")


# ═══════════════════════════════════════
# 文档上传 & 摄入 (Modular)
# ═══════════════════════════════════════

class IngestResponse(BaseModel):
    run_id: str
    filename: str
    pages: int
    chunks: int
    details: List[Dict[str, Any]]

@app.post("/api/v1/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文件并触发模块化摄入流水线"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {', '.join(sorted(SUPPORTED_EXTS))}",
        )

    # 保存文件
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 执行流水线
    result = pipeline.ingest(save_path)
    return result

class ChatInput(BaseModel):
    question: str

@app.post("/api/v1/chat/invoke")
async def chat_invoke(input_data: ChatInput):
    """模块化 RAG 问答"""
    answer = pipeline.chat(input_data.question)
    return {"output": answer}


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "debug_mode": settings.debug_pipeline,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "mysql_uri": settings.mysql_uri.split("@")[-1],
        "milvus": {
            "uri": settings.milvus_uri,
            "collection": settings.milvus_collection,
            "index_type": settings.milvus_index_type,
            "metric_type": settings.milvus_metric_type,
            "vector_dim": settings.milvus_vector_dim,
            "hnsw": {
                "m": settings.hnsw_m,
                "ef_construction": settings.hnsw_ef_construction,
                "ef_search": settings.hnsw_ef_search,
            },
        },
        "retriever": {
            "top_k": settings.retriever_top_k,
            "score_semantics": {
                "raw": "milvus_native_score",
                "normalized": "higher_is_better",
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
