# RAG-Memo — 模块化多模态文档 RAG 系统

基于 **模块化 Pipeline** 设计的检索增强生成系统，每个阶段可独立调试和调优。

## 架构

```
app/
├── core/
│   ├── config.py              # 全局配置 (.env)
│   └── models.py              # MySQL ORM 模型 (文档/切片/流水线记录)
├── db/
│   ├── milvus.py              # Milvus collection/schema/index 管理
│   └── mysql.py               # MySQL 连接管理
├── services/                  # ⭐ 六大可插拔模块及流水线编排
│   ├── chunker.py             # 文本切片 (递归切分)
│   ├── context.py             # 上下文工程 (格式化/截断)
│   ├── embedder.py            # 向量化 + Milvus 入库 (HNSW)
│   ├── loader.py              # 文档加载 (PDF/图片/文本)
│   ├── pipeline.py            # 流水线编排器 (串联所有模块)
│   ├── reranker.py            # 重排序 (Qwen3-Reranker-8B)
│   └── retriever.py           # 向量召回 (HNSW + COSINE)
├── schemas/
│   └── __init__.py            # Pydantic 数据模型
├── utils/
│   ├── embeddings.py          # Embedding 模型工厂
│   ├── llm.py                 # LLM 客户端
│   ├── logger.py              # Rich 终端可视化 + Debug 开关
│   └── ocr.py                 # RapidOCR 文字识别
├── data/                      # 文档存放目录
│   └── uploads/               # 上传文件目录
└── main.py                    # FastAPI 入口
```

## 模块化流水线

### 摄入流程
```
文件 → [Loader] → [Chunker] → [Embedder] → Milvus
```

### 问答流程
```
Query → [Retriever] → [Reranker] → [Context] → [LLM] → Answer
```

每个模块的运行状态、耗时、输入/输出摘要均可在终端实时查看。

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 框架 | FastAPI |
| Embedding | `qwen3-embedding-8b` (OpenAI 兼容，支持分批并发调用) |
| Reranker | `qwen3-reranker-8b` |
| LLM | `qwen3-30b-a3b` |
| 向量存储 | Milvus (显式 schema/index 管理, 默认 HNSW + COSINE) |
| 关系存储 | MySQL 8.0 (流水线记录/文档元数据) |
| PDF 解析 | PyMuPDF + RapidOCR |
| 终端可视化 | Rich + Loguru |

## 快速开始

```bash
# 1. 启动基础设施 (Milvus + MySQL)
cd docker && docker-compose up -d

# 2. 复制配置文件并填写 API Key
cp .env.example .env

# 3. 安装依赖
uv sync

# 4. 启动服务
uv run python -m app.main
```

## HNSW / Milvus 可调参数

可通过 `.env` 调整以下索引与搜索参数：

```bash
MILVUS_INDEX_TYPE=HNSW
MILVUS_METRIC_TYPE=COSINE
MILVUS_VECTOR_DIM=1024
HNSW_M=16
HNSW_EF_CONSTRUCTION=256
HNSW_EF_SEARCH=128

# Embedding 并发参数
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_CONCURRENCY=4
EMBEDDING_PARALLEL_THRESHOLD=64

# Chunking 并发参数
CHUNKING_MAX_CONCURRENCY=4
CHUNKING_PARALLEL_THRESHOLD=8
```

当一次摄入的页面/文档段较多时，Chunker 会在达到 `CHUNKING_PARALLEL_THRESHOLD` 后按文档粒度并发切片，以减少大批量文档同时进入系统时的切分等待时间。

当单次摄入的 chunk 数较多时，系统会按 `EMBEDDING_BATCH_SIZE` 分批，并在达到 `EMBEDDING_PARALLEL_THRESHOLD` 后以最多 `EMBEDDING_MAX_CONCURRENCY` 的并发度调用 embedding 接口，以提升大文档/大量切片的向量化速度。

当前检索阶段会同时保留两类分数：

- `ann_raw_score`: Milvus 原始返回分数
- `recall_score`: 统一后的归一化相关度分数，约定为 `higher_is_better`

并可通过 `GET /api/v1/health` 查看当前 HNSW / score 相关配置。

同时支持文档级索引管理与检索过滤：

- `GET /api/v1/documents`：查看已摄入文档
- `DELETE /api/v1/documents/{document_id}`：删除文档及对应向量
- `POST /api/v1/documents/{document_id}/reindex`：按 document_id 重建索引
- `POST /api/v1/chat/invoke` 支持可选过滤参数：`document_ids` / `filename` / `source`

切片元信息会尽量保留定位字段：

- `document_title`：文件标题
- `section_title`：当前片段所在标题
- `heading_path`：层级标题路径，例如 `第1章 > 1.2 背景 > 1.2.1 定义`
- `heading_titles`：层级标题数组
- `source_start_index`：该片段在原页面/原文本中的起始偏移

> 标题提取目前采用启发式规则（Markdown 标题、编号标题、中文“第X章/节”标题、短标题行）。

## Debug 模式

通过 `.env` 中的 `DEBUG_PIPELINE` 控制终端输出详细程度：

**开启 (`DEBUG_PIPELINE=true`)**：显示每个模块的详细面板输出
```
▶ Starting Module: Loader
🟢 Module: loader     | Status: success  | Time: 40ms | ...
╭──────────────── loader Detail Output ────────────────╮
│ Understanding Climate Change ...                      │
╰──────────────────────────────────────────────────────╯
```

**关闭 (`DEBUG_PIPELINE=false`)**：仅显示精简摘要
```
🟢 Module: loader     | Status: success  | Time: 40ms | ...
```

## 测试

```bash
# 运行模块化流水线测试
uv run pytest tests/test_pipeline.py
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文件并触发摄入流水线 |
| GET  | `/api/v1/documents` | 查看已摄入文档与 chunk 数 |
| DELETE | `/api/v1/documents/{document_id}` | 删除文档及对应向量索引 |
| POST | `/api/v1/documents/{document_id}/reindex` | 重建指定文档索引 |
| POST | `/api/v1/chat/invoke` | 模块化 RAG 问答，支持文档范围过滤 |
| GET  | `/api/v1/health` | 健康检查 (含 debug / HNSW 配置) |
