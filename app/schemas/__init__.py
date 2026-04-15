from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime

class ModuleResult(BaseModel):
    """模块运行结果"""
    module_name: str
    status: str = "success"
    duration_ms: float = 0.0
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    data: Optional[Any] = None  # 阶段性的实际数据（切片、检索结果等）

class PipelineRunSchema(BaseModel):
    """流水线运行记录摘要"""
    run_id: str
    query: str
    answer: Optional[str] = None
    total_duration_ms: float = 0.0
    created_at: datetime
    module_results: List[ModuleResult] = Field(default_factory=list)

class DocumentSchema(BaseModel):
    """文档记录"""
    id: int
    filename: str
    file_type: str
    status: str
    created_at: datetime

class ChunkSchema(BaseModel):
    """切片记录"""
    content: str
    chunk_index: int
    page_num: Optional[int] = None
    score: Optional[float] = None # 用于召回结果
    source: Optional[str] = None
