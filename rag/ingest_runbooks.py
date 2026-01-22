from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag.chunking import load_markdown_chunks
from rag.retriever import DEFAULT_ARTIFACT_DIR, get_embedder, persist_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest runbooks into FAISS index.")
    parser.add_argument(
        "--runbook-dir",
        type=Path,
        default=Path("data/sample_runbooks"),
        help="Directory containing markdown runbooks.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory to store FAISS index and chunk metadata.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="mock",
        help="SentenceTransformer model name or 'mock' for offline deterministic embeddings.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=120,
        help="Maximum words per chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.info("Loading runbooks from %s", args.runbook_dir)
    chunks = load_markdown_chunks(args.runbook_dir, max_words=args.max_words)
    logging.info("Loaded %s chunks", len(chunks))

    embedder = get_embedder(args.embedding_model)
    logging.info("Using embedding model %s (dim=%s)", embedder.name, embedder.dim)
    persist_index(chunks, embedder, artifact_dir=args.artifact_dir)
    logging.info("Artifacts saved to %s", args.artifact_dir)


if __name__ == "__main__":
    main()
