from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from app.core.config import settings
import uuid

class Chunker:
    """文本切片模块 (Module: Chunker)"""
    
    def __init__(
        self, 
        chunk_size: int = None, 
        chunk_overlap: int = None,
        strategy: str = "recursive"
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy

    def run(self, documents: List[LCDocument]) -> List[LCDocument]:
        """执行切分流程"""
        if self.strategy == "recursive":
            return self._split_recursive(documents)
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")

    def _split_recursive(self, documents: List[LCDocument]) -> List[LCDocument]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ".", ";", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        
        # 为切片注入元数据
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            # 如果 collection schema 定义了 chunk_id (int64)，则注入一个整数
            # 否则不要随意注入会导致 DataNotMatchException
            # chunk.metadata["chunk_id"] = idx 
            
        return chunks
