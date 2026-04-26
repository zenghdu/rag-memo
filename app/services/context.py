from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document as LCDocument
from app.utils.logger import logger

class ContextBuilder:
    """上下文工程模块 (Module: Context)"""
    
    def __init__(self, max_tokens: int = 4096, template: str = None):
        self.max_tokens = max_tokens
        self.template = template or (
            "【检索到的文档片段】\n{context}\n\n"
            "请根据上述文档片段回答用户问题：\n{question}"
        )

    def run(self, query: str, scored_docs: List[Tuple[LCDocument, float]]) -> str:
        """格式化上下文文本"""
        logger.debug(f"Building context for query: {query}")
        
        if not scored_docs:
            return "（未检索到相关文档内容）"

        # 格式化文档片段
        parts = []
        for i, (doc, score) in enumerate(scored_docs, 1):
            meta = doc.metadata
            source = meta.get("filename", "unknown")
            page = meta.get("page", "?")
            document_title = meta.get("document_title") or source
            heading_path = meta.get("heading_path")
            raw_score = meta.get("recall_score_raw")
            if raw_score is None:
                score_text = f"相关度(归一化): {score:.4f}"
            else:
                score_text = f"相关度(归一化): {score:.4f} | 原始分数: {raw_score:.4f}"
            location = f"标题: {document_title} | 来源: {source} | 页码: {page}"
            if heading_path:
                location += f" | 层级标题: {heading_path}"
            parts.append(f"[{i}] {location} | {score_text}\n{doc.page_content}")
            
        context_str = "\n\n---\n\n".join(parts)
        
        # 填充模板
        # TODO: 真正应用 token 截断逻辑 (tiktoken)
        return self.template.format(context=context_str, question=query)
