"""Vector store providers behind a common interface.

Supports ChromaDB (default, local persistent) and Pinecone (remote, requires API key).
"""

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
# Pinecone provider
# ---------------------------------------------------------------------------


_PINECONE_DOCUMENT_KEY = "_document"


class PineconeStore:
    """Remote vector store backed by Pinecone.

    Document text is stored in metadata under the ``_document`` key since
    Pinecone does not have a separate document field.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str,
        namespace: str,
    ) -> None:
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._namespace = namespace

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Upsert vectors with document text stored in metadata."""
        if not ids:
            return
        vectors = []
        for i, doc_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas else {}
            meta[_PINECONE_DOCUMENT_KEY] = documents[i]
            vectors.append({"id": doc_id, "values": embeddings[i], "metadata": meta})
        self._index.upsert(vectors=vectors, namespace=self._namespace)

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[QueryResult]:
        """Similarity search via the Pinecone query API."""
        kwargs: dict = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self._namespace,
        }
        if where is not None:
            kwargs["filter"] = where
        results = self._index.query(**kwargs)

        query_results: list[QueryResult] = []
        for match in results["matches"]:
            metadata = dict(match.get("metadata", {}))
            document = metadata.pop(_PINECONE_DOCUMENT_KEY, "")
            query_results.append(
                QueryResult(
                    id=match["id"],
                    document=document,
                    metadata=metadata,
                    distance=1.0 - match["score"],
                )
            )
        return query_results

    def delete(self, ids: list[str]) -> None:
        """Remove vectors by their IDs."""
        if not ids:
            return
        self._index.delete(ids=ids, namespace=self._namespace)

    def get(self, ids: list[str]) -> list[StoredDocument]:
        """Retrieve documents by their IDs via the Pinecone fetch API."""
        if not ids:
            return []
        results = self._index.fetch(ids=ids, namespace=self._namespace)
        stored: list[StoredDocument] = []
        for doc_id in ids:
            vec = results["vectors"].get(doc_id)
            if vec is not None:
                metadata = dict(vec.get("metadata", {}))
                document = metadata.pop(_PINECONE_DOCUMENT_KEY, "")
                stored.append(StoredDocument(id=doc_id, document=document, metadata=metadata))
        return stored

    def count(self) -> int:
        """Return total vector count from index stats for this namespace."""
        stats = self._index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
        return ns_stats.get("vector_count", 0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vectorstore(
    config: Settings,
    collection_name: str,
    chroma_path: Path | None = None,
) -> VectorStore:
    """Instantiate the appropriate :class:`VectorStore` based on *config*.

    Parameters
    ----------
    config:
        Application settings (vectorstore backend selection + Pinecone creds).
    collection_name:
        Name of the vector collection (typically the KB UUID).
    chroma_path:
        Filesystem path for ChromaDB persistence.  Required when the
        ``chromadb`` backend is selected.

    Raises:
        ValueError: If required configuration is missing for the chosen backend.
    """
    if config.vectorstore == "chromadb":
        if chroma_path is None:
            raise ValueError("chroma_path is required for the chromadb backend")
        return ChromaDBStore(
            persist_path=chroma_path,
            collection_name=collection_name,
        )
    elif config.vectorstore == "pinecone":
        if not all([config.pinecone_api_key, config.pinecone_index, config.pinecone_environment]):
            raise ValueError(
                "Pinecone requires AGENT_KB_PINECONE_API_KEY, "
                "AGENT_KB_PINECONE_INDEX, and AGENT_KB_PINECONE_ENVIRONMENT"
            )
        return PineconeStore(
            api_key=config.pinecone_api_key,  # type: ignore[arg-type]
            index_name=config.pinecone_index,  # type: ignore[arg-type]
            environment=config.pinecone_environment,  # type: ignore[arg-type]
            namespace=collection_name,
        )
    else:
        raise ValueError(f"Unknown vectorstore provider: {config.vectorstore}")
