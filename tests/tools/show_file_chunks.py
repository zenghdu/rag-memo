import json
from app.db.milvus import MilvusVectorStore
from rich.console import Console
from rich.panel import Panel

console = Console()
store = MilvusVectorStore()
collection = store.ensure_collection()
collection.load()

# 我们以刚才测试过的那个文件为例，看看它被切分成了几个 chunk，内容到底是什么
filename = "上海保险专业中介机构合规管理暂行办法.doc"
expr = f"filename == '{filename}'"

results = collection.query(
    expr=expr,
    output_fields=["document_id", "chunk_index", "filename", "text", "page_num", "metadata_json"],
    limit=100
)

# 按 chunk_index 排序
results.sort(key=lambda x: x.get("chunk_index", 0))

console.print(f"\n[bold green]📊 找到了 {len(results)} 个属于文件 '{filename}' 的切片 (Chunks)[/]\n")

for res in results:
    title = f"[bold cyan]{res.get('filename')}[/] | Chunk: [bold yellow]{res.get('chunk_index')}[/] | Page: {res.get('page_num')}"
    text = res.get('text', '')
    
    # 格式化元数据
    try:
        meta = json.loads(res.get('metadata_json', '{}'))
        meta_str = "\n".join([f"  - {k}: {v}" for k, v in meta.items()])
    except:
        meta_str = str(res.get('metadata_json'))

    content = f"[bold magenta]=== 文本内容 (长度: {len(text)} 字符) ===[/]\n{text}\n\n[bold magenta]=== 元数据 ===[/]\n{meta_str}"
    
    console.print(Panel(content, title=title, border_style="blue"))
