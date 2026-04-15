from datetime import datetime
from typing import Any, Optional
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass

class Document(Base):
    """原始文档记录"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_type = Column(String(20))
    file_hash = Column(String(64), unique=True)  # 用于去重
    status = Column(String(20), default="pending")  # pending, processing, completed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    """切片详情"""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    page_num = Column(Integer)
    token_count = Column(Integer)
    metadata_json = Column(JSON)
    milvus_id = Column(String(100)) # 关联 Milvus 中的 ID
    
    document = relationship("Document", back_populates="chunks")

class PipelineRun(Base):
    """RAG 执行流水线记录"""
    __tablename__ = "pipeline_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True)
    query = Column(Text, nullable=False)
    answer = Column(Text)
    total_duration_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    module_logs = relationship("ModuleLog", back_populates="run", cascade="all, delete-orphan")

class ModuleLog(Base):
    """单模块执行日志"""
    __tablename__ = "module_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"))
    module_name = Column(String(50), nullable=False)  # loader, chunker, etc.
    status = Column(String(20)) # success, error
    duration_ms = Column(Float)
    input_summary = Column(JSON)
    output_summary = Column(JSON)
    error_msg = Column(Text)
    
    run = relationship("PipelineRun", back_populates="module_logs")
