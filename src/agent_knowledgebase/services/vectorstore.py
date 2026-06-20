"""Vector store provider — ChromaDB only (v0.13.0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """A single result from a similarity search."""

    id: str
    document: str
    metadata: dict
    distance: float


@dataclass
class StoredDocument:
    """A document retrieved by ID from the store."""

    id: str
    document: str
    metadata: dict


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Common interface that all vector store providers must satisfy."""

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Upsert embeddings, documents, and optional metadata."""
        ...

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[QueryResult]:
        """Similarity search returning results sorted by distance (ascending)."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""
        ...

    def get(self, ids: list[str]) -> list[StoredDocument]:
        """Retrieve documents by their IDs."""
        ...

    def count(self) -> int:
        """Return the number of stored vectors."""
        ...


# ---------------------------------------------------------------------------
# ChromaDB provider
# ---------------------------------------------------------------------------


class ChromaDBStore:
    """Local vector store backed by ChromaDB with persistent storage.

    Each knowledgebase gets its own collection (collection name = kb_id).
    """

    def __init__(self, persist_path: Path, collection_name: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Upsert embeddings + documents + metadata into the collection."""
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[QueryResult]:
        """Similarity search returning :class:`QueryResult` list sorted by distance."""
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)

        query_results: list[QueryResult] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                query_results.append(
                    QueryResult(
                        id=doc_id,
                        document=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i],
                    )
                )
        return query_results

    def delete(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""
        if not ids:
            return
        self._collection.delete(ids=ids)

    def get(self, ids: list[str]) -> list[StoredDocument]:
        """Retrieve documents by their IDs."""
        if not ids:
            return []
        results = self._collection.get(ids=ids, include=["documents", "metadatas"])
        stored: list[StoredDocument] = []
        for i, doc_id in enumerate(results["ids"]):
            stored.append(
                StoredDocument(
                    id=doc_id,
                    document=results["documents"][i],
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
            )
        return stored

    def count(self) -> int:
        """Return the number of stored vectors in the collection."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vectorstore(
    config: Settings,
    collection_name: str,
    chroma_path: Path | None = None,
) -> VectorStore:
    """Instantiate a ChromaDBStore for the given KB collection."""
    if chroma_path is None:
        raise ValueError("chroma_path is required for the chromadb backend")
    return ChromaDBStore(persist_path=chroma_path, collection_name=collection_name)
