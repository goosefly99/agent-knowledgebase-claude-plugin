"""Phase A tests: ChromadbBackend filter passthrough (Defect 2 fix).

Pins three behaviours:
  1. ChromadbBackend.query(filters={...}) no longer raises NotImplementedError.
  2. A valid filter dict is piped to the vectorstore via where= (merged with
     the existing kb_id filter injected by QueryOrchestrator.query).
  3. A structurally invalid filter dict (non-string key, non-primitive value)
     is rejected before the chromadb call (does NOT reach chromadb).

Note on the validator: chromadb's where= accepts a Mongo-style dict, not a
SQL string. We validate the dict structurally (string keys, primitive scalar
values) rather than via the SQL AST validator (which operates on strings and
is not applicable to chromadb filter dicts).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(test_config: Settings) -> KnowledgebaseService:
    """KnowledgebaseService with mocked vectorstore, embedder, and ingestion."""
    svc = KnowledgebaseService(test_config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.return_value = [[0.1, 0.2, 0.3]]
    svc._embedder_instance.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    svc._embedder_instance.model_name = "filter-test-embedder"
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.return_value = []
    svc._ingestion = mock_ingestion
    return svc


def _make_mock_vs() -> MagicMock:
    """Return a mock VectorStore that records query calls."""
    vs = MagicMock()
    vs.query = MagicMock(return_value=[])  # empty results is fine
    vs.add = MagicMock(return_value=None)
    vs.delete = MagicMock(return_value=None)
    vs.count = MagicMock(return_value=0)
    return vs


# ---------------------------------------------------------------------------
# Defect 2 fix: no longer raises NotImplementedError
# ---------------------------------------------------------------------------


class TestFilterNoLongerRaises:
    """query() and search() with filters must NOT raise NotImplementedError."""

    def test_query_with_filter_does_not_raise(self, test_config: Settings) -> None:
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-no-raise-query")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        # Should not raise NotImplementedError.
        result = backend.query(
            kb_id=kb.id,
            text="test query",
            top_k=5,
            filters={"embedding_model": "some-model"},
        )
        # Result shape: list of dicts.
        assert isinstance(result, list)

    def test_search_with_filter_does_not_raise(self, test_config: Settings) -> None:
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-no-raise-search")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        # Should not raise NotImplementedError.
        result = backend.search(
            kb_id=kb.id,
            text="test search",
            top_k=5,
            filters={"source_type": "file"},
        )
        assert isinstance(result, list)

    def test_query_without_filter_still_works(self, test_config: Settings) -> None:
        """Passing filters=None (the default) must keep working."""
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-none-query")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        result = backend.query(kb_id=kb.id, text="hello", top_k=3)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Filter dict is piped to the vectorstore via where=
# ---------------------------------------------------------------------------


class TestFilterPassedToVectorstore:
    """The filter dict reaches the vectorstore's where= parameter."""

    def test_filter_keys_merged_into_where_dict(self, test_config: Settings) -> None:
        """Extra filter keys are merged into the vectorstore where= call."""
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-merged-query")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)
        extra_filter = {"embedding_model": "my-model"}

        backend.query(
            kb_id=kb.id,
            text="test",
            top_k=3,
            filters=extra_filter,
        )

        # vs.query must have been called with a where= that includes the filter key.
        assert mock_vs.query.called, "vectorstore.query was never called"
        _, call_kwargs = mock_vs.query.call_args
        where_used = call_kwargs.get("where", {})
        assert "embedding_model" in where_used, (
            f"filter key 'embedding_model' was not propagated into vs.query where=; "
            f"got where={where_used!r}"
        )
        assert where_used["embedding_model"] == "my-model"

    def test_filter_dict_is_merged_with_kb_id_filter(self, test_config: Settings) -> None:
        """The kb_id filter and the extra filter keys coexist in where=."""
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-kb-id-coexist")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        backend.query(
            kb_id=kb.id,
            text="coexist test",
            top_k=2,
            filters={"source_type": "file"},
        )

        assert mock_vs.query.called
        _, call_kwargs = mock_vs.query.call_args
        where_used = call_kwargs.get("where", {})
        # Both kb_id AND the extra filter key must be present.
        assert "kb_id" in where_used, f"kb_id filter was dropped; got where={where_used!r}"
        assert "source_type" in where_used, (
            f"filter key 'source_type' not in where; got where={where_used!r}"
        )

    def test_no_filter_uses_only_kb_id_where(self, test_config: Settings) -> None:
        """Without filters the existing kb_id-only where= dict is used."""
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-none-where")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        backend.query(kb_id=kb.id, text="no filter", top_k=2)

        assert mock_vs.query.called
        _, call_kwargs = mock_vs.query.call_args
        where_used = call_kwargs.get("where", {})
        # Only kb_id; no extra keys injected.
        assert "kb_id" in where_used
        assert where_used["kb_id"] == kb.id


# ---------------------------------------------------------------------------
# Structurally invalid filter dicts are rejected before chromadb is called
# ---------------------------------------------------------------------------


class TestInvalidFilterRejected:
    """Invalid filter dicts raise ValueError before reaching chromadb."""

    def test_non_string_key_rejected(self, test_config: Settings) -> None:
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-invalid-key")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        with pytest.raises((ValueError, TypeError)):
            backend.query(
                kb_id=kb.id,
                text="bad filter",
                top_k=2,
                filters={123: "value"},  # type: ignore[dict-item]
            )
        # Vectorstore must NOT have been called.
        assert not mock_vs.query.called, "chromadb was reached despite invalid filter"

    def test_non_primitive_value_rejected(self, test_config: Settings) -> None:
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-invalid-value")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        # A nested dict as a value is not a primitive scalar.
        with pytest.raises((ValueError, TypeError)):
            backend.query(
                kb_id=kb.id,
                text="bad filter",
                top_k=2,
                filters={"key": {"nested": "dict"}},
            )
        assert not mock_vs.query.called, "chromadb was reached despite invalid filter"

    def test_empty_string_key_rejected(self, test_config: Settings) -> None:
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-empty-key")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        with pytest.raises((ValueError, TypeError)):
            backend.query(
                kb_id=kb.id,
                text="bad filter",
                top_k=2,
                filters={"": "value"},
            )
        assert not mock_vs.query.called

    def test_valid_filter_allows_chromadb_call(self, test_config: Settings) -> None:
        """A well-formed filter dict allows the vs.query call to proceed."""
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("filter-valid-passes")
        mock_vs = _make_mock_vs()
        svc._get_vectorstore = lambda _: mock_vs  # type: ignore[assignment]

        backend = ChromadbBackend(test_config, service=svc)

        # All primitive types should be valid.
        backend.query(
            kb_id=kb.id,
            text="valid filter",
            top_k=2,
            filters={"str_key": "str_val", "int_key": 42, "bool_key": True},
        )
        assert mock_vs.query.called, "valid filter should have allowed vs.query call"
