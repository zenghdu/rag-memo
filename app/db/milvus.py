"""Milvus 兼容性补丁 & 连接配置"""

from pymilvus import connections
from langchain_milvus import Milvus

# ── langchain-milvus 0.3.x + pymilvus 2.6.x 兼容性修复 ──
# MilvusClient 内部连接别名未注册到 ORM connections 模块，
# 导致 Collection(using=alias) 抛出 ConnectionNotExistException。
_original_col_getter = Milvus.col.fget


def _patched_col_getter(self):  # type: ignore
    alias = self.alias
    if not connections.has_connection(alias):
        connections.connect(alias=alias, **self._connection_args)
    return _original_col_getter(self)


Milvus.col = property(_patched_col_getter, Milvus.col.fset)  # type: ignore
# ── 修复结束 ──
