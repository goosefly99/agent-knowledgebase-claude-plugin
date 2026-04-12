"""Tests for agent_knowledgebase.services.query (QueryOrchestrator)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.models import PageType, WikiPage
from agent_knowledgebase.services.query import QueryOrchestrator, SearchResult, _normalize_scores
from agent_knowledgebase.services.vectorstore import QueryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KB_ID = "test-kb-001"


def _make_wiki_page(
    title: str,
    content: str,
    page_type: PageType = PageType.entity,
    **kwargs: object,
) -> WikiPage:
    """Helper to build a WikiPage with sensible defaults."""
    return WikiPage(
        kb_id=KB_ID,
        title=title,
        content=content,
        page_type=page_type,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture()
def mock_vectorstore() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture()
def mock_wiki() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def orchestrator(
    mock_vectorstore: MagicMock,
    mock_embedder: MagicMock,
    mock_wiki: MagicMock,
) -> QueryOrchestrator:
    return QueryOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        wiki=mock_wiki,
    )


# ---------------------------------------------------------------------------
# _normalize_scores helper
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_empty_list(self) -> None:
        assert _normalize_scores([]) == []

    def test_single_item(self) -> None:
        results = [SearchResult(content="a", source_id="1", source_type="chunk", score=0.5)]
        _normalize_scores(results)
        assert results[0].score == 1.0

    def test_uniform_scores(self) -> None:
        results = [
            SearchResult(content="a", source_id="1", source_type="chunk", score=0.7),
            SearchResult(content="b", source_id="2", source_type="chunk", score=0.7),
        ]
        _normalize_scores(results)
        assert all(r.score == 1.0 for r in results)

    def test_diverse_scores(self) -> None:
        results = [
            SearchResult(content="a", source_id="1", source_type="chunk", score=0.2),
            SearchResult(content="b", source_id="2", source_type="chunk", score=0.6),
            SearchResult(content="c", source_id="3", source_type="chunk", score=1.0),
        ]
        _normalize_scores(results)
        assert results[0].score == pytest.approx(0.0)
        assert results[1].score == pytest.approx(0.5)
        assert results[2].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Semantic query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_embeds_text_and_searches(
        self,
        orchestrator: QueryOrchestrator,
        mock_embedder: MagicMock,
        mock_vectorstore: MagicMock,
    ) -> None:
        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document="chunk one", metadata={"kb_id": KB_ID}, distance=0.1),
            QueryResult(id="c2", document="chunk two", metadata={"kb_id": KB_ID}, distance=0.3),
        ]

        results = orchestrator.query("test question", KB_ID)

        mock_embedder.embed_query.assert_called_once_with("test question")
        mock_vectorstore.query.assert_called_once_with(
            [0.1, 0.2, 0.3], top_k=10, where={"kb_id": KB_ID}
        )

        assert len(results) == 2
        assert all(r.source_type == "chunk" for r in results)
        assert results[0].source_id == "c1"
        assert results[0].content == "chunk one"
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.7)

    def test_empty_results(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
    ) -> None:
        mock_vectorstore.query.return_value = []
        results = orchestrator.query("nothing matches", KB_ID)
        assert results == []

    def test_score_clamped_to_0_1(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
    ) -> None:
        """Distances > 1 or < 0 should still produce clamped scores."""
        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document="d", metadata={}, distance=1.5),
            QueryResult(id="c2", document="d", metadata={}, distance=-0.1),
        ]
        results = orchestrator.query("q", KB_ID)
        assert results[0].score == 0.0
        assert results[1].score == 1.0

    def test_respects_top_k(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
    ) -> None:
        mock_vectorstore.query.return_value = []
        orchestrator.query("q", KB_ID, top_k=5)
        mock_vectorstore.query.assert_called_once_with(
            [0.1, 0.2, 0.3], top_k=5, where={"kb_id": KB_ID}
        )


# ---------------------------------------------------------------------------
# Keyword / FTS search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_fts_returns_ranked_results(
        self,
        orchestrator: QueryOrchestrator,
        mock_wiki: MagicMock,
    ) -> None:
        mock_wiki.search.return_value = [
            _make_wiki_page("Page A", "content alpha"),
            _make_wiki_page("Page B", "content beta"),
            _make_wiki_page("Page C", "content gamma"),
        ]

        results = orchestrator.search("alpha", KB_ID)

        mock_wiki.search.assert_called_once_with("alpha", KB_ID)
        assert len(results) == 3
        assert all(r.source_type == "wiki_page" for r in results)

        # First result should have the highest score.
        assert results[0].score == pytest.approx(1.0)
        assert results[0].content == "content alpha"
        assert results[0].metadata["title"] == "Page A"

        # Scores decrease with position.
        assert results[0].score > results[1].score > results[2].score

    def test_empty_results(
        self,
        orchestrator: QueryOrchestrator,
        mock_wiki: MagicMock,
    ) -> None:
        mock_wiki.search.return_value = []
        results = orchestrator.search("nonexistent", KB_ID)
        assert results == []

    def test_metadata_populated(
        self,
        orchestrator: QueryOrchestrator,
        mock_wiki: MagicMock,
    ) -> None:
        mock_wiki.search.return_value = [
            _make_wiki_page(
                "Tagged Page",
                "body",
                page_type=PageType.concept,
                tags=["python", "testing"],
            ),
        ]
        results = orchestrator.search("body", KB_ID)
        assert results[0].metadata["page_type"] == "concept"
        assert results[0].metadata["tags"] == ["python", "testing"]

    def test_respects_top_k(
        self,
        orchestrator: QueryOrchestrator,
        mock_wiki: MagicMock,
    ) -> None:
        mock_wiki.search.return_value = [
            _make_wiki_page(f"Page {i}", f"content {i}") for i in range(10)
        ]
        results = orchestrator.search("content", KB_ID, top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Hybrid query
# ---------------------------------------------------------------------------


class TestHybridQuery:
    def test_combines_both_sources(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document="chunk about python", metadata={}, distance=0.1),
        ]
        mock_wiki.search.return_value = [
            _make_wiki_page("Python Guide", "wiki about python"),
        ]

        results = orchestrator.hybrid_query("python", KB_ID, top_k=10)

        assert len(results) == 2
        source_types = {r.source_type for r in results}
        assert source_types == {"chunk", "wiki_page"}

    def test_correct_score_weighting(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        """With only vector results, combined score = vector_weight * normalized_score."""
        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document="only vector", metadata={}, distance=0.2),
        ]
        mock_wiki.search.return_value = []

        results = orchestrator.hybrid_query(
            "q", KB_ID, top_k=10, vector_weight=0.7, fts_weight=0.3
        )

        assert len(results) == 1
        # Single result normalizes to 1.0, then multiplied by vector_weight.
        assert results[0].score == pytest.approx(0.7)

    def test_deduplication_merges_scores(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        """When both sources return the same content, scores are merged."""
        shared_content = "shared content about testing"

        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document=shared_content, metadata={}, distance=0.2),
        ]
        mock_wiki.search.return_value = [
            _make_wiki_page("Test Page", shared_content),
        ]

        results = orchestrator.hybrid_query(
            "testing", KB_ID, top_k=10, vector_weight=0.7, fts_weight=0.3
        )

        # Should be deduplicated to a single result with merged score.
        assert len(results) == 1
        # Both normalize to 1.0 (single items), so combined = 0.7*1.0 + 0.3*1.0 = 1.0.
        assert results[0].score == pytest.approx(1.0)

    def test_respects_top_k_limit(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        mock_vectorstore.query.return_value = [
            QueryResult(id=f"c{i}", document=f"chunk {i}", metadata={}, distance=0.1 * i)
            for i in range(1, 6)
        ]
        mock_wiki.search.return_value = [
            _make_wiki_page(f"Page {i}", f"wiki {i}") for i in range(1, 6)
        ]

        results = orchestrator.hybrid_query("q", KB_ID, top_k=3)
        assert len(results) == 3

    def test_custom_weights(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        """FTS-heavy weighting should rank FTS-only results higher."""
        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document="vector only", metadata={}, distance=0.5),
        ]
        mock_wiki.search.return_value = [
            _make_wiki_page("FTS Page", "fts only"),
        ]

        # FTS-heavy: vector_weight=0.1, fts_weight=0.9
        results = orchestrator.hybrid_query(
            "q", KB_ID, top_k=10, vector_weight=0.1, fts_weight=0.9
        )

        assert len(results) == 2
        # The FTS-only result should have score 0.9 * 1.0 = 0.9,
        # vector-only should have score 0.1 * 1.0 = 0.1.
        fts_result = next(r for r in results if r.source_type == "wiki_page")
        vec_result = next(r for r in results if r.source_type == "chunk")
        assert fts_result.score > vec_result.score
        assert fts_result.score == pytest.approx(0.9)
        assert vec_result.score == pytest.approx(0.1)

    def test_over_fetches_for_merging(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        """hybrid_query should pass top_k*2 to underlying query and search."""
        mock_vectorstore.query.return_value = []
        mock_wiki.search.return_value = []

        orchestrator.hybrid_query("q", KB_ID, top_k=5)

        # Verify vectorstore was called with top_k=10 (5*2).
        call_args = mock_vectorstore.query.call_args
        assert call_args[1]["top_k"] == 10 or call_args[0][1] == 10

    def test_results_sorted_descending(
        self,
        orchestrator: QueryOrchestrator,
        mock_vectorstore: MagicMock,
        mock_wiki: MagicMock,
    ) -> None:
        mock_vectorstore.query.return_value = [
            QueryResult(id="c1", document="low score chunk", metadata={}, distance=0.9),
            QueryResult(id="c2", document="high score chunk", metadata={}, distance=0.1),
        ]
        mock_wiki.search.return_value = [
            _make_wiki_page("High FTS", "high fts content"),
            _make_wiki_page("Low FTS", "low fts content"),
        ]

        results = orchestrator.hybrid_query("q", KB_ID, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
