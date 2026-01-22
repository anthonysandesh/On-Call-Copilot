from rag.chunking import chunk_markdown


def test_chunk_ids_are_deterministic():
    text = "# Title\n\n## Section\nContent line one.\nMore content.\n"
    chunks1 = chunk_markdown(text, "sample.md", max_words=5)
    chunks2 = chunk_markdown(text, "sample.md", max_words=5)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]
    assert chunks1[0].metadata["heading_path"] == ["Title", "Section"]
    assert chunks1[0].metadata["chunk_index"] == "0-0"
