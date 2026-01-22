from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rag.chunking import Chunk

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - dependency warning
    faiss = None
    logging.warning("FAISS not available: %s", exc)


DEFAULT_ARTIFACT_DIR = Path("artifacts")
CHUNKS_FILE = "chunks.jsonl"
INDEX_FILE = "faiss.index"
META_FILE = "index_meta.json"


class EmbeddingModel:
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class MockEmbeddingModel(EmbeddingModel):
    def __init__(self, dim: int = 64):
        self.name = "mock"
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            # Stable seed from text hash for deterministic embeddings
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors.append(rng.standard_normal(self.dim))
        return np.vstack(vectors).astype("float32")


class SentenceTransformerEmbedding(EmbeddingModel):  # pragma: no cover - thin wrapper
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise ImportError("sentence-transformers is required for this embedder") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")


def get_embedder(model_name: Optional[str] = None) -> EmbeddingModel:
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "mock")
    if model_name.lower() in {"mock", "mock-embedding"} or model_name.lower().startswith(
        "mock"
    ):
        return MockEmbeddingModel()
    return SentenceTransformerEmbedding(model_name)


@dataclass
class Retriever:
    embedder: EmbeddingModel
    index: object
    chunks: List[Chunk]

    @classmethod
    def load(cls, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> "Retriever":
        meta_path = artifact_dir / META_FILE
        chunks_path = artifact_dir / CHUNKS_FILE
        index_path = artifact_dir / INDEX_FILE

        if not meta_path.exists() or not chunks_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"Artifacts not found in {artifact_dir}, run `make ingest`.")

        meta = json.loads(meta_path.read_text())
        embedder = get_embedder(meta.get("embedding_model"))
        if faiss is None:
            raise ImportError("faiss is required to load the index")
        index = faiss.read_index(str(index_path))

        chunks: List[Chunk] = []
        with chunks_path.open() as f:
            for line in f:
                obj = json.loads(line)
                chunks.append(Chunk(id=obj["id"], text=obj["text"], metadata=obj["metadata"]))

        return cls(embedder=embedder, index=index, chunks=chunks)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        if faiss is None:
            raise ImportError("faiss is required for retrieval")
        query_vec = self.embedder.embed([query])
        scores, idxs = self.index.search(query_vec, min(k, len(self.chunks)))
        scored: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            scored.append((self.chunks[idx], float(score)))
        return scored


def persist_index(
    chunks: List[Chunk],
    embedder: EmbeddingModel,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> None:
    if faiss is None:
        raise ImportError("faiss is required to build the index")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed(texts).astype("float32")
    index = faiss.IndexFlatL2(embedder.dim)
    index.add(vectors)

    faiss.write_index(index, str(artifact_dir / INDEX_FILE))

    meta = {
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "chunk_count": len(chunks),
    }
    (artifact_dir / META_FILE).write_text(json.dumps(meta, indent=2))

    with (artifact_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps({"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata})
                + "\n"
            )
