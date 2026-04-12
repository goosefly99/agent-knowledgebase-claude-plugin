"""Tests for agent_knowledgebase.services.vectorstore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.vectorstore import (
    ChromaDBStore,
    PineconeStore,
    QueryResult,
    StoredDocument,
    VectorStore,
    create_vectorstore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simple deterministic embeddings for testing. Each vector is 4-dimensional.
_DIM = 4
_EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],  # close to _EMBEDDINGS[0]
]


# ---------------------------------------------------------------------------
# ChromaDB — real local instance via tmp_path
# ---------------------------------------------------------------------------


class TestChromaDBStore:
    """Integration tests using a real ChromaDB persistent client."""

    @pytest.fixture()
    def store(self, tmp_path: Path) -> ChromaDBStore:
        """Return a ChromaDBStore backed by a temporary directory."""
        return ChromaDBStore(persist_path=tmp_path / "chroma", collection_name="test-kb")

    def test_satisfies_protocol(self, store: ChromaDBStore) -> None:
        assert isinstance(store, VectorStore)

    def test_count_empty(self, store: ChromaDBStore) -> None:
        assert store.count() == 0

    def test_add_and_count(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["a", "b"],
            embeddings=_EMBEDDINGS[:2],
            documents=["doc a", "doc b"],
        )
        assert store.count() == 2

    def test_add_with_metadata(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["a"],
            embeddings=[_EMBEDDINGS[0]],
            documents=["doc a"],
            metadatas=[{"source": "test"}],
        )
        results = store.get(ids=["a"])
        assert len(results) == 1
        assert results[0].metadata["source"] == "test"

    def test_add_empty_list(self, store: ChromaDBStore) -> None:
        """Adding an empty list should be a no-op."""
        store.add(ids=[], embeddings=[], documents=[])
        assert store.count() == 0

    def test_add_upsert_overwrites(self, store: ChromaDBStore) -> None:
        """Adding an existing ID should update the document."""
        store.add(
            ids=["a"],
            embeddings=[_EMBEDDINGS[0]],
            documents=["original"],
        )
        store.add(
            ids=["a"],
            embeddings=[_EMBEDDINGS[0]],
            documents=["updated"],
        )
        assert store.count() == 1
        results = store.get(ids=["a"])
        assert results[0].document == "updated"

    def test_get_returns_stored_documents(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["x", "y"],
            embeddings=_EMBEDDINGS[:2],
            documents=["doc x", "doc y"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
        )
        results = store.get(ids=["x", "y"])
        assert len(results) == 2
        assert all(isinstance(r, StoredDocument) for r in results)
        docs_by_id = {r.id: r for r in results}
        assert docs_by_id["x"].document == "doc x"
        assert docs_by_id["y"].metadata["k"] == "v2"

    def test_get_empty_ids(self, store: ChromaDBStore) -> None:
        assert store.get(ids=[]) == []

    def test_delete_removes_documents(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["a", "b", "c"],
            embeddings=_EMBEDDINGS[:3],
            documents=["doc a", "doc b", "doc c"],
        )
        assert store.count() == 3
        store.delete(ids=["b"])
        assert store.count() == 2
        remaining = store.get(ids=["a", "c"])
        assert len(remaining) == 2

    def test_delete_empty_ids(self, store: ChromaDBStore) -> None:
        """Deleting an empty list should be a no-op."""
        store.add(ids=["a"], embeddings=[_EMBEDDINGS[0]], documents=["doc a"])
        store.delete(ids=[])
        assert store.count() == 1

    def test_query_returns_results_sorted_by_distance(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["a", "b", "c"],
            embeddings=_EMBEDDINGS[:3],
            documents=["doc a", "doc b", "doc c"],
        )
        # Query with a vector close to 'a'
        results = store.query(embedding=_EMBEDDINGS[0], top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)
        # Closest result should be 'a' (distance ~0)
        assert results[0].id == "a"
        # Distances should be non-decreasing
        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_query_top_k_limits_results(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["a", "b", "c"],
            embeddings=_EMBEDDINGS[:3],
            documents=["doc a", "doc b", "doc c"],
        )
        results = store.query(embedding=_EMBEDDINGS[0], top_k=1)
        assert len(results) == 1

    def test_query_empty_store(self, store: ChromaDBStore) -> None:
        results = store.query(embedding=_EMBEDDINGS[0], top_k=5)
        assert results == []

    def test_query_returns_documents_and_metadata(self, store: ChromaDBStore) -> None:
        store.add(
            ids=["a"],
            embeddings=[_EMBEDDINGS[0]],
            documents=["doc a"],
            metadatas=[{"source": "unit-test"}],
        )
        results = store.query(embedding=_EMBEDDINGS[0], top_k=1)
        assert results[0].document == "doc a"
        assert results[0].metadata["source"] == "unit-test"

    def test_query_similarity_ordering(self, store: ChromaDBStore) -> None:
        """A vector very close to 'a' should rank 'a' first."""
        store.add(
            ids=["a", "b", "c", "d"],
            embeddings=_EMBEDDINGS,
            documents=["doc a", "doc b", "doc c", "doc d"],
        )
        # _EMBEDDINGS[3] is [0.9, 0.1, 0.0, 0.0] — very close to _EMBEDDINGS[0]
        results = store.query(embedding=_EMBEDDINGS[3], top_k=4)
        # 'a' and 'd' should be the top two results (closest to each other)
        top_ids = {results[0].id, results[1].id}
        assert top_ids == {"a", "d"}

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Data survives across separate ChromaDBStore instances."""
        chroma_dir = tmp_path / "chroma_persist"
        store1 = ChromaDBStore(persist_path=chroma_dir, collection_name="persist-test")
        store1.add(
            ids=["p1"],
            embeddings=[_EMBEDDINGS[0]],
            documents=["persistent doc"],
        )
        # Create a new instance pointing to the same directory
        store2 = ChromaDBStore(persist_path=chroma_dir, collection_name="persist-test")
        assert store2.count() == 1
        results = store2.get(ids=["p1"])
        assert results[0].document == "persistent doc"


# ---------------------------------------------------------------------------
# Pinecone — fully mocked
# ---------------------------------------------------------------------------


class TestPineconeStore:
    """Unit tests with a fully mocked Pinecone client."""

    @pytest.fixture()
    def mock_index(self) -> MagicMock:
        """Return a mock Pinecone index."""
        return MagicMock()

    @pytest.fixture()
    def store(self, mock_index: MagicMock) -> PineconeStore:
        """Return a PineconeStore with mocked internals."""
        with patch("agent_knowledgebase.services.vectorstore.PineconeStore.__init__", lambda s, **kw: None):
            s = PineconeStore.__new__(PineconeStore)
        s._index = mock_index
        s._namespace = "test-ns"
        return s

    def test_satisfies_protocol(self, store: PineconeStore) -> None:
        assert isinstance(store, VectorStore)

    def test_add_upserts_vectors_with_document_in_metadata(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        store.add(
            ids=["a", "b"],
            embeddings=_EMBEDDINGS[:2],
            documents=["doc a", "doc b"],
            metadatas=[{"source": "s1"}, {"source": "s2"}],
        )
        mock_index.upsert.assert_called_once()
        call_kwargs = mock_index.upsert.call_args
        vectors = call_kwargs.kwargs["vectors"]
        assert len(vectors) == 2
        # Document text should be stored in metadata under '_document'
        assert vectors[0]["metadata"]["_document"] == "doc a"
        assert vectors[0]["metadata"]["source"] == "s1"
        assert vectors[1]["metadata"]["_document"] == "doc b"
        assert call_kwargs.kwargs["namespace"] == "test-ns"

    def test_add_without_metadata(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        store.add(
            ids=["a"],
            embeddings=[_EMBEDDINGS[0]],
            documents=["doc a"],
        )
        vectors = mock_index.upsert.call_args.kwargs["vectors"]
        assert vectors[0]["metadata"] == {"_document": "doc a"}

    def test_add_empty_list(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        store.add(ids=[], embeddings=[], documents=[])
        mock_index.upsert.assert_not_called()

    def test_query_calls_index_query(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        mock_index.query.return_value = {
            "matches": [
                {"id": "a", "score": 0.95, "metadata": {"_document": "doc a", "source": "s1"}},
                {"id": "b", "score": 0.80, "metadata": {"_document": "doc b"}},
            ]
        }
        results = store.query(embedding=_EMBEDDINGS[0], top_k=5)

        mock_index.query.assert_called_once_with(
            vector=_EMBEDDINGS[0],
            top_k=5,
            include_metadata=True,
            namespace="test-ns",
        )
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].document == "doc a"
        assert results[0].metadata == {"source": "s1"}
        # Distance = 1 - score for cosine
        assert results[0].distance == pytest.approx(0.05)
        assert results[1].distance == pytest.approx(0.20)

    def test_query_with_filter(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        mock_index.query.return_value = {"matches": []}
        store.query(embedding=_EMBEDDINGS[0], top_k=3, where={"source": "test"})
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == {"source": "test"}

    def test_delete_calls_index_delete(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        store.delete(ids=["a", "b"])
        mock_index.delete.assert_called_once_with(ids=["a", "b"], namespace="test-ns")

    def test_delete_empty_list(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        store.delete(ids=[])
        mock_index.delete.assert_not_called()

    def test_get_fetches_and_extracts_documents(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        mock_index.fetch.return_value = {
            "vectors": {
                "a": {"metadata": {"_document": "doc a", "source": "s1"}},
                "b": {"metadata": {"_document": "doc b"}},
            }
        }
        results = store.get(ids=["a", "b"])
        mock_index.fetch.assert_called_once_with(ids=["a", "b"], namespace="test-ns")
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].document == "doc a"
        assert results[0].metadata == {"source": "s1"}
        assert results[1].id == "b"
        assert results[1].document == "doc b"

    def test_get_empty_ids(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        assert store.get(ids=[]) == []
        mock_index.fetch.assert_not_called()

    def test_get_skips_missing_ids(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        mock_index.fetch.return_value = {
            "vectors": {
                "a": {"metadata": {"_document": "doc a"}},
            }
        }
        results = store.get(ids=["a", "missing"])
        assert len(results) == 1
        assert results[0].id == "a"

    def test_count_returns_namespace_vector_count(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        mock_index.describe_index_stats.return_value = {
            "namespaces": {
                "test-ns": {"vector_count": 42},
                "other-ns": {"vector_count": 100},
            }
        }
        assert store.count() == 42

    def test_count_returns_zero_for_unknown_namespace(
        self, store: PineconeStore, mock_index: MagicMock
    ) -> None:
        mock_index.describe_index_stats.return_value = {"namespaces": {}}
        assert store.count() == 0


# ---------------------------------------------------------------------------
# create_vectorstore factory
# ---------------------------------------------------------------------------


class TestCreateVectorstore:
    """Factory function tests."""

    def test_creates_chromadb_store(self, tmp_path: Path) -> None:
        config = Settings(
            vectorstore="chromadb",
            chroma_path=tmp_path / "chroma",
        )
        store = create_vectorstore(config, collection_name="test-kb")
        assert isinstance(store, ChromaDBStore)

    def test_creates_pinecone_store(self) -> None:
        config = Settings(
            vectorstore="pinecone",
            pinecone_api_key="pk-test",
            pinecone_index="my-index",
            pinecone_environment="us-east-1",
        )
        mock_pc_cls = MagicMock()
        mock_pc_cls.return_value.Index.return_value = MagicMock()
        with patch.dict("sys.modules", {"pinecone": MagicMock(Pinecone=mock_pc_cls)}):
            store = create_vectorstore(config, collection_name="test-kb")
        assert isinstance(store, PineconeStore)

    def test_pinecone_missing_api_key_raises(self) -> None:
        config = Settings(
            vectorstore="pinecone",
            pinecone_api_key=None,
            pinecone_index="my-index",
            pinecone_environment="us-east-1",
        )
        with pytest.raises(ValueError, match="Pinecone requires"):
            create_vectorstore(config, collection_name="test-kb")

    def test_pinecone_missing_index_raises(self) -> None:
        config = Settings(
            vectorstore="pinecone",
            pinecone_api_key="pk-test",
            pinecone_index=None,
            pinecone_environment="us-east-1",
        )
        with pytest.raises(ValueError, match="Pinecone requires"):
            create_vectorstore(config, collection_name="test-kb")

    def test_pinecone_missing_environment_raises(self) -> None:
        config = Settings(
            vectorstore="pinecone",
            pinecone_api_key="pk-test",
            pinecone_index="my-index",
            pinecone_environment=None,
        )
        with pytest.raises(ValueError, match="Pinecone requires"):
            create_vectorstore(config, collection_name="test-kb")
