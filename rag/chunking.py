from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, object]


def stable_hash(value: str, length: int = 12) -> str:
    """Return a deterministic, short hash for IDs."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]


def slugify_path(path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path.lower()).strip("-")
    return slug or "runbook"


def _split_sections(lines: Sequence[str]) -> Iterable[Tuple[List[str], List[str]]]:
    """Yield (heading_path, lines) tuples for each markdown section."""
    heading_path: List[str] = []
    buffer: List[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if buffer:
                yield heading_path.copy(), buffer
                buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_path = heading_path[: level - 1] + [title]
            continue
        buffer.append(line)

    if buffer:
        yield heading_path.copy(), buffer


def _chunk_text(text: str, max_words: int = 120) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        start = end
    return chunks


def chunk_markdown(text: str, source_path: str, max_words: int = 120) -> List[Chunk]:
    """Chunk markdown text with heading context. Deterministic IDs."""
    lines = text.splitlines()
    base_slug = slugify_path(Path(source_path).stem)

    chunks: List[Chunk] = []
    section_index = 0
    for heading_path, section_lines in _split_sections(lines):
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        heading_str = " > ".join(heading_path) if heading_path else Path(source_path).stem
        for idx, chunk_body in enumerate(_chunk_text(section_text, max_words=max_words)):
            chunk_index = f"{section_index}-{idx}"
            chunk_id = f"rbk-{base_slug}-{stable_hash(source_path + heading_str + chunk_index + chunk_body)}"
            combined_text = f"{heading_str}\n{chunk_body}".strip()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=combined_text,
                    metadata={
                        "source": str(Path(source_path)),
                        "heading_path": heading_path,
                        "chunk_index": chunk_index,
                    },
                )
            )
        section_index += 1
    return chunks


def load_markdown_chunks(base_dir: Path, max_words: int = 120) -> List[Chunk]:
    """Load and chunk all markdown files under base_dir."""
    all_chunks: List[Chunk] = []
    for path in sorted(base_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(text, source_path=str(path), max_words=max_words))
    return all_chunks
