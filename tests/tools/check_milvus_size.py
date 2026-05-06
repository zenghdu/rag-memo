from app.db.milvus import MilvusVectorStore
from pymilvus import utility, Collection

store = MilvusVectorStore()
collection = store.ensure_collection()
collection.flush()
print(f"Collection name: {collection.name}")
print(f"Total entities (rows): {collection.num_entities}")

# Try to get unique filenames if there are entities
if collection.num_entities > 0:
    results = collection.query(expr="", output_fields=["filename"], limit=collection.num_entities)
    filenames = set(res.get("filename") for res in results if res.get("filename"))
    print(f"Unique files in DB: {filenames}")
