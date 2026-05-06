from typing import Dict, List, Optional, Tuple
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from app.core.config import settings


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
        if not documents:
            return []

        parallel_threshold = max(1, settings.chunking_parallel_threshold)
        max_concurrency = max(1, settings.chunking_max_concurrency)

        if len(documents) < parallel_threshold or max_concurrency == 1:
            chunk_groups = [self._split_one_document(document) for document in documents]
        else:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(documents))) as executor:
                chunk_groups = list(executor.map(self._split_one_document, documents))

        chunks: List[LCDocument] = []
        chunk_index = 0
        for group in chunk_groups:
            for local_index, chunk in enumerate(group):
                chunk.metadata["chunk_index"] = chunk_index
                chunk.metadata["chunk_local_index"] = local_index
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_one_document(self, document: LCDocument) -> List[LCDocument]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ".", ";", " ", ""],
            add_start_index=True,
        )

        source_chunks = splitter.split_documents([document])
        headings = self._extract_headings(document.page_content)
        resolved_chunks: List[LCDocument] = []
        for chunk in source_chunks:
            start_index = int(chunk.metadata.get("start_index", 0) or 0)
            section_info = self._resolve_section_info(headings, start_index)
            chunk.metadata["source_start_index"] = start_index
            chunk.metadata["document_title"] = chunk.metadata.get(
                "document_title",
                document.metadata.get("document_title") or document.metadata.get("filename", "unknown"),
            )
            chunk.metadata.update(section_info)
            resolved_chunks.append(chunk)

        return resolved_chunks

    def _extract_headings(self, text: str) -> List[Dict[str, object]]:
        headings: List[Dict[str, object]] = []
        offset = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.strip()
            if line:
                heading = self._parse_heading(line)
                if heading:
                    headings.append(
                        {
                            "start": offset,
                            "title": heading["title"],
                            "level": heading["level"],
                            "raw": line,
                        }
                    )
            offset += len(raw_line)
        return headings

    def _parse_heading(self, line: str) -> Optional[Dict[str, object]]:
        markdown = re.match(r"^(#{1,6})\s+(.+)$", line)
        if markdown:
            return {"level": len(markdown.group(1)), "title": markdown.group(2).strip()}

        numbered = re.match(r"^(\d+(?:\.\d+){0,5})[\.、\-\s:：]+(.+)$", line)
        if numbered:
            return {
                "level": numbered.group(1).count(".") + 1,
                "title": f"{numbered.group(1)} {numbered.group(2).strip()}",
            }

        chapter = re.match(r"^(第[一二三四五六七八九十百千万0-9]+[章节篇部卷])\s*(.+)?$", line)
        if chapter:
            suffix = chapter.group(2).strip() if chapter.group(2) else ""
            title = f"{chapter.group(1)} {suffix}".strip()
            level = 2 if "节" in chapter.group(1) else 1
            return {"level": level, "title": title}

        if self._looks_like_short_heading(line):
            return {"level": 1, "title": line}

        return None

    def _looks_like_short_heading(self, line: str) -> bool:
        if len(line) > 60:
            return False
        if any(p in line for p in ["。", "；", ";", ".", "?", "？", "!", "！"]):
            return False
        if len(line.split()) > 12:
            return False
        return True

    def _resolve_section_info(self, headings: List[Dict[str, object]], start_index: int) -> Dict[str, object]:
        if not headings:
            return {
                "section_title": None,
                "section_level": None,
                "heading_path": None,
                "heading_titles": [],
            }

        stack: List[Dict[str, object]] = []
        for heading in headings:
            heading_start = int(heading["start"])
            if heading_start > start_index:
                break
            level = int(heading["level"])
            stack = [item for item in stack if int(item["level"]) < level]
            stack.append(heading)

        if not stack:
            return {
                "section_title": None,
                "section_level": None,
                "heading_path": None,
                "heading_titles": [],
            }

        heading_titles = [str(item["title"]) for item in stack]
        current = stack[-1]
        return {
            "section_title": str(current["title"]),
            "section_level": int(current["level"]),
            "heading_path": " > ".join(heading_titles),
            "heading_titles": heading_titles,
        }
