"""Pin ChromaDBStore mutability invariants for v0.13.0.

The plugin contract is that knowledgebases are MUTABLE: ingesting
the same id twice updates the row in place; deleting an id removes
both the vector and any subsequent query result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.services.vectorstore import ChromaDBStore


@pytest.fixture()
def store(tmp_path: Path) -> ChromaDBStore:
    return ChromaDBStore(persist_path=tmp_path / "chroma", collection_name="mutability_test")


def test_add_inserts_new_documents(store: ChromaDBStore) -> None:
    store.add(
        ids=["a", "b"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["alpha", "beta"],
        metadatas=[{"k": "1"}, {"k": "2"}],
    )
    assert store.count() == 2


def test_add_with_existing_id_updates_in_place(store: ChromaDBStore) -> None:
    store.add(
        ids=["a"],
        embeddings=[[0.1, 0.2]],
        documents=["original"],
        metadatas=[{"version": "1"}],
    )
    assert store.count() == 1
    store.add(
        ids=["a"],
        embeddings=[[0.9, 0.8]],
        documents=["updated"],
        metadatas=[{"version": "2"}],
    )
    assert store.count() == 1  # not inserted twice
    fetched = store.get(["a"])
    assert fetched[0].document == "updated"
    assert fetched[0].metadata["version"] == "2"


def test_delete_removes_documents(store: ChromaDBStore) -> None:
    store.add(
        ids=["a", "b", "c"],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        documents=["x", "y", "z"],
        metadatas=[{"k": "a"}, {"k": "b"}, {"k": "c"}],
    )
    store.delete(["b"])
    assert store.count() == 2
    assert store.get(["b"]) == []


def test_query_reflects_post_delete_state(store: ChromaDBStore) -> None:
    store.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["matches-a", "matches-b"],
        metadatas=[{"k": "a"}, {"k": "b"}],
    )
    store.delete(["a"])
    results = store.query(embedding=[1.0, 0.0], top_k=5)
    ids = [r.id for r in results]
    assert "a" not in ids
    assert "b" in ids
