from __future__ import annotations

from typing import List, Sequence, Tuple

from rag.chunking import Chunk
from serving.schemas import TriageResponse


def missing_citations(response: TriageResponse, retrieved: Sequence[Tuple[Chunk, float]]) -> List[str]:
    retrieved_ids = {chunk.id for chunk, _ in retrieved}
    return [cid for cid in response.citations if cid not in retrieved_ids]


def citation_coverage(response: TriageResponse) -> float:
    total = len(response.remediation_steps)
    if total == 0:
        return 1.0
    with_cite = sum(1 for step in response.remediation_steps if step.citation_ids)
    return with_cite / total
