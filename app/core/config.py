"""项目配置 — 通过 .env 文件或环境变量注入"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """从 .env 读取配置，支持环境变量覆盖"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Embedding ----
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"

    # ---- LLM ----
    llm_model: str = "qwen/qwen3-30b-a3b"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"

    # ---- Milvus ----
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_documents"
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"
    milvus_vector_dim: int = 1024
    hnsw_m: int = 16
    hnsw_ef_construction: int = 256
    hnsw_ef_search: int = 128

    # ---- MySQL ----
    mysql_uri: str = "mysql+pymysql://user:password@localhost:3306/rag_memo"

    # ---- Debug ----
    debug_pipeline: bool = True

    # ---- Reranker ----
    reranker_model: str = "qwen/qwen3-reranker-8b"
    reranker_api_key: str = ""
    reranker_base_url: str = "https://api.openai.com/v1"

    # ---- RAG ----
    chunk_size: int = 800
    chunk_overlap: int = 150
    retriever_top_k: int = 5


settings = Settings()
