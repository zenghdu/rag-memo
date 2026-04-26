from typing import Any, Dict
import hashlib
import time
import uuid
from pathlib import Path

from app.core.models import Document as DBDocument, Chunk as DBChunk
from app.db.mysql import SessionLocal
from app.schemas import ModuleResult
from app.utils.logger import (
    print_module_start,
    print_module_summary,
    print_final_answer,
    PipelineProgress,
)

# 模块引入
from app.services.loader import Loader
from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.retriever import Retriever
from app.services.reranker import Reranker
from app.services.context import ContextBuilder
from app.utils.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class PipelineService:
    """流水线执行服务 (Module Orchestrator)"""

    def __init__(self):
        self.loader = Loader()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()

    def ingest(self, file_path: Path) -> Dict[str, Any]:
        """文档摄入全流程: Load -> Chunk -> Embedder"""
        run_id = str(uuid.uuid4())
        results = []
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        with SessionLocal() as db:
            db_document = db.query(DBDocument).filter(DBDocument.file_hash == file_hash).one_or_none()
            if db_document is None:
                db_document = DBDocument(
                    filename=file_path.name,
                    file_path=str(file_path),
                    file_type=file_path.suffix.lower().lstrip("."),
                    file_hash=file_hash,
                    status="processing",
                )
                db.add(db_document)
                db.commit()
                db.refresh(db_document)
            else:
                db_document.filename = file_path.name
                db_document.file_path = str(file_path)
                db_document.file_type = file_path.suffix.lower().lstrip(".")
                db_document.status = "processing"
                db.query(DBChunk).filter(DBChunk.document_id == db_document.id).delete()
                db.commit()

            # 1. Loader
            print_module_start("Loader")
            t0 = time.time()
            docs = self.loader.run(file_path)
            t_load = (time.time() - t0) * 1000

            res_load = ModuleResult(
                module_name="loader",
                duration_ms=t_load,
                input_summary={"file": file_path.name},
                output_summary={"pages": len(docs), "preview": docs[0].page_content[:200] if docs else ""},
            )
            print_module_summary(**res_load.model_dump(exclude={"data"}))
            results.append(res_load)

            # 2. Chunker
            print_module_start("Chunker")
            t1 = time.time()
            chunks = self.chunker.run(docs)
            for idx, chunk in enumerate(chunks):
                chunk.metadata["document_id"] = db_document.id
                chunk.metadata["chunk_index"] = idx
            t_chunk = (time.time() - t1) * 1000

            res_chunk = ModuleResult(
                module_name="chunker",
                duration_ms=t_chunk,
                input_summary={"pages": len(docs)},
                output_summary={"chunks": len(chunks), "preview": chunks[0].page_content[:200] if chunks else ""},
            )
            print_module_summary(**res_chunk.model_dump(exclude={"data"}))
            results.append(res_chunk)

            # 3. Embedder
            print_module_start("Embedder")
            t2 = time.time()
            self.embedder.vector_store.delete_by_document_id(db_document.id)
            emb_res = self.embedder.run(chunks)
            t_emb = (time.time() - t2) * 1000

            res_emb = ModuleResult(
                module_name="embedder",
                duration_ms=t_emb,
                input_summary={"chunks": len(chunks)},
                output_summary={"inserted": emb_res["count"], "ids": str(emb_res.get("ids", []))},
            )
            print_module_summary(**res_emb.model_dump(exclude={"data"}))
            results.append(res_emb)

            db.query(DBChunk).filter(DBChunk.document_id == db_document.id).delete()
            for idx, chunk in enumerate(chunks):
                db.add(
                    DBChunk(
                        document_id=db_document.id,
                        content=chunk.page_content,
                        chunk_index=chunk.metadata.get("chunk_index", idx),
                        page_num=chunk.metadata.get("page"),
                        token_count=max(1, len(chunk.page_content) // 4),
                        metadata_json=chunk.metadata,
                        milvus_id=emb_res["all_ids"][idx] if idx < len(emb_res.get("all_ids", [])) else None,
                    )
                )
            db_document.status = "completed"
            db.commit()

        return {
            "run_id": run_id,
            "filename": file_path.name,
            "pages": len(docs),
            "chunks": len(chunks),
            "details": results,
        }

    def chat(self, query: str) -> str:
        """RAG 问答全流程: Retriever -> Reranker -> Context -> LLM"""
        run_id = str(uuid.uuid4())
        results = []

        with PipelineProgress(query) as p:
            # 1. Retriever (Recall)
            print_module_start("Retriever")
            t_r0 = time.time()
            scored_docs = self.retriever.run(query)
            t_retrieval = (time.time() - t_r0) * 1000

            res_ret = ModuleResult(
                module_name="retriever",
                duration_ms=t_retrieval,
                input_summary={"query": query},
                output_summary={
                    "recalled": len(scored_docs),
                    "preview": f"Score: {scored_docs[0][1]:.4f} | Content: {scored_docs[0][0].page_content[:100]}" if scored_docs else "Empty",
                },
            )
            print_module_summary(**res_ret.model_dump(exclude={"data"}))
            results.append(res_ret)

            # 2. Reranker
            print_module_start("Reranker")
            t_rr0 = time.time()
            reranked_docs = self.reranker.run(query, scored_docs)
            t_rerank = (time.time() - t_rr0) * 1000

            res_rr = ModuleResult(
                module_name="reranker",
                duration_ms=t_rerank,
                input_summary={"query": query, "recalled": len(scored_docs)},
                output_summary={
                    "reranked": len(reranked_docs),
                    "preview": f"Score: {reranked_docs[0][1]:.4f} | Content: {reranked_docs[0][0].page_content[:100]}" if reranked_docs else "Empty",
                },
            )
            print_module_summary(**res_rr.model_dump(exclude={"data"}))
            results.append(res_rr)

            # 3. Context Engineering
            print_module_start("Context")
            t_c0 = time.time()
            context_text = self.context_builder.run(query, reranked_docs)
            t_context = (time.time() - t_c0) * 1000

            res_ctx = ModuleResult(
                module_name="context",
                duration_ms=t_context,
                input_summary={"reranked": len(reranked_docs)},
                output_summary={"preview": context_text[:200] + "..."},
            )
            print_module_summary(**res_ctx.model_dump(exclude={"data"}))
            results.append(res_ctx)

            # 4. LLM Generation
            print_module_start("LLM")
            llm = get_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个文档助手，请基于提供的上下文回答。"),
                ("human", "{question_with_context}"),
            ])

            t_l0 = time.time()
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"question_with_context": context_text})
            t_llm = (time.time() - t_l0) * 1000

            res_llm = ModuleResult(
                module_name="llm",
                duration_ms=t_llm,
                input_summary={"tokens_approx": len(context_text) // 2},
                output_summary={"preview": answer[:100]},
            )
            print_module_summary(**res_llm.model_dump(exclude={"data"}))
            results.append(res_llm)

            print_final_answer(answer, run_id)
            return answer
