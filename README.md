# RAG-Memo — 模块化多模态文档 RAG 系统

基于 **模块化 Pipeline** 设计的检索增强生成系统，每个阶段可独立调试和调优。

## 架构

```
app/
├── core/
│   ├── config.py              # 全局配置 (.env)
│   └── models.py              # MySQL ORM 模型 (文档/切片/流水线记录)
├── modules/                   # ⭐ 六大可插拔模块
│   ├── loader.py              # 文档加载 (PDF/图片/文本)
│   ├── chunker.py             # 文本切片 (递归切分)
│   ├── embedder.py            # 向量化 + Milvus 入库 (HNSW)
│   ├── retriever.py           # 向量召回 (HNSW + COSINE)
│   ├── reranker.py            # 重排序 (Qwen3-Reranker-8B)
│   └── context.py             # 上下文工程 (格式化/截断)
├── db/
│   ├── milvus.py              # Milvus 兼容性补丁
│   └── mysql.py               # MySQL 连接管理
├── services/
│   ├── embeddings.py          # Embedding 模型工厂
│   ├── rag_chain.py           # LLM 工厂
│   └── pipeline_service.py    # 流水线编排器 (串联所有模块)
├── schemas/
│   └── __init__.py            # Pydantic 数据模型
├── utils/
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
| Embedding | `qwen3-embedding-8b` (OpenAI 兼容) |
| Reranker | `qwen3-reranker-8b` |
| LLM | `qwen3-30b-a3b` |
| 向量存储 | Milvus (HNSW 索引, COSINE 距离) |
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
uv run python tests/reconstruction/test_modular_pipeline.py
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文件并触发摄入流水线 |
| POST | `/api/v1/chat/invoke` | 模块化 RAG 问答 |
| GET  | `/api/v1/health` | 健康检查 (含 debug 模式状态) |
