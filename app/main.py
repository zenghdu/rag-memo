# app/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# 1. 初始化 FastAPI
app = FastAPI(
    title="金融合规 RAG 系统 API",
    version="1.0",
    description="基于 FastAPI + LangChain + Milvus + MySQL 的混合检索问答"
)

@app.get("/")
async def redirect_root_to_docs():
    """默认跳转到 Swagger UI 文档页"""
    return RedirectResponse("/docs")

# ==========================================
# 2. 占位：这里未来会替换成我们从 app.services.rag_svc 导入的真实 RAG Chain
# 这里先写一个最简单的 Dummy Chain 用于测试 LangServe
# ==========================================
prompt = ChatPromptTemplate.from_template("你说一个关于【{topic}】的金融笑话。")

# 模拟一个处理逻辑 (未来这里是 ChatOpenAI 等大模型)
def dummy_llm(text: str) -> str:
    return f"这是一个关于 {text} 的笑话：为什么程序员不喜欢炒股？因为他们讨厌 Bug（熊市）。"

dummy_chain = prompt | RunnableLambda(dummy_llm)


# ==========================================
# 3. 核心魔法：把 Chain 挂载为 API
# ==========================================
add_routes(
    app,
    dummy_chain,
    path="/api/v1/chat", # 前端调用路径
)

if __name__ == "__main__":
    import uvicorn
    # 本地启动命令：uv run python -m app.main
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
