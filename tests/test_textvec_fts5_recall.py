"""Phase B smoke recall test: FTS5+BM25 returns expected matches.

Ingest a small curated corpus (15 chunks with known content), run 8
queries with known expected matches, assert >= 80% recall@10.

Designed to run in < 2s (no external services, in-memory-style).

NOTE: This test does NOT pull from
pipeline_mcp_data/collections/curated/agent-kb-textvec-sources--curated.json
(heavy 16-transcript fixture). That fixture is used by bin/benchmark_recall.py
instead. This CI-fast inline test uses a compact synthetic corpus.
"""

from __future__ import annotations

import uuid
from pathlib import Path


from agent_knowledgebase.backends.textvec_backend import TextvecBackend
from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.database import Database


# ---------------------------------------------------------------------------
# Small curated corpus with known matching properties
# ---------------------------------------------------------------------------

_CORPUS = [
    # id, content
    ("c01", "Neural networks learn representations from data using backpropagation."),
    ("c02", "Attention mechanisms allow models to focus on relevant parts of the input sequence."),
    ("c03", "Transformers use self-attention and feed-forward layers to process tokens."),
    ("c04", "Gradient descent optimizes neural network weights by minimizing a loss function."),
    (
        "c05",
        "Retrieval augmented generation combines document retrieval with language model generation.",
    ),
    ("c06", "Vector embeddings encode semantic meaning as dense numerical representations."),
    (
        "c07",
        "BM25 is a lexical ranking function based on term frequency and document length normalization.",
    ),
    ("c08", "Knowledge graphs represent entities and their relationships as nodes and edges."),
    ("c09", "Fine-tuning adapts a pretrained model to a specific downstream task."),
    ("c10", "Chunking splits long documents into smaller pieces for embedding and retrieval."),
    ("c11", "SQLite full text search FTS5 supports Porter stemming and Unicode tokenization."),
    (
        "c12",
        "The retrieval pipeline indexes documents and retrieves relevant chunks at query time.",
    ),
    ("c13", "Embeddings from large language models capture contextual word relationships."),
    ("c14", "Sparse retrieval methods like TF-IDF and BM25 work well on keyword-heavy queries."),
    ("c15", "Dense retrieval uses neural embeddings to find semantically similar documents."),
]

# Queries with expected matching chunk ids (by content overlap).
# A match is counted if the expected chunk appears in top-10 results.
#
# IMPORTANT: FTS5 uses AND-of-tokens logic by default — a document must
# contain ALL query tokens (after Porter stemming) to match. Queries and
# expected IDs are designed to reflect this:
# - Each query term must appear in the expected document.
# - Expected IDs only include documents that contain ALL query terms.
_QUERIES = [
    # c03 contains both "self-attention" (→ attention) and "transformers" (→ transform).
    ("attention transformer", ["c03"]),
    # c01 contains "neural", "networks", "backpropagation" (all after stemming).
    ("neural networks backpropagation", ["c01"]),
    # c04 contains "gradient", "descent" (both terms).
    ("gradient descent optimize", ["c04"]),
    # c07 contains "BM25", "ranking", "term", "frequency".
    ("BM25 ranking frequency", ["c07"]),
    # c06 contains "vector", "embeddings", "semantic", "dense".
    ("vector embeddings semantic dense", ["c06"]),
    # c05 contains "retrieval", "augmented", "generation".
    ("retrieval augmented generation", ["c05"]),
    # c11 contains "FTS5", "porter", "stemming", "SQLite".
    ("FTS5 porter stemming SQLite", ["c11"]),
    # c08 contains "knowledge", "graphs", "entities", "relationships".
    ("knowledge graphs entities relationships", ["c08"]),
    # c09 contains "pretrained", "model", "task", "downstream".
    ("pretrained model task downstream", ["c09"]),
    # c12 contains "retrieval", "pipeline", "indexes", "chunks".
    ("retrieval pipeline indexes chunks", ["c12"]),
    # c14 contains "sparse", "retrieval", "BM25".
    ("sparse retrieval BM25", ["c14"]),
    # c15 contains "dense", "retrieval", "neural", "embeddings".
    ("dense retrieval neural embeddings", ["c15"]),
    # c10 contains "chunking", "splits", "documents".
    ("chunking splits documents", ["c10"]),
    # c13 contains "embeddings", "language", "models".
    ("embeddings language models", ["c13"]),
    # c02 contains "attention", "mechanisms", "models".
    ("attention mechanisms models", ["c02"]),
]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_fresh_backend(tmp_path: Path, kb_id: str) -> TextvecBackend:
    saves = tmp_path / "saves"
    saves.mkdir(exist_ok=True)
    settings = Settings(saves_dir=saves)
    safe = sanitize_kb_dir_name(kb_id)
    db_path = settings.kb_db_path(safe)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path=db_path)
    db._conn.execute(
        "INSERT OR IGNORE INTO knowledgebases (id, name, description, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, kb_id, "", "{}", "2024-01-01", "2024-01-01"),
    )
    db._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb_id}", kb_id, "file", "/tmp/recall.txt", "ingested"),
    )
    db._conn.commit()
    return TextvecBackend(settings, service=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recall_at_10_above_threshold(tmp_path: Path) -> None:
    """smoke recall test: >= 80% of expected chunks appear in top-10 results."""
    kb_id = f"recall-{uuid.uuid4().hex[:8]}"
    backend = _build_fresh_backend(tmp_path, kb_id)

    # Ingest corpus.
    docs = [
        {
            "id": cid,
            "content": text,
            "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}", "source_type": "file"},
        }
        for cid, text in _CORPUS
    ]
    backend.index(kb_id=kb_id, documents=docs)

    total_expected = 0
    total_hit = 0

    for query_text, expected_ids in _QUERIES:
        results = backend.search(kb_id=kb_id, text=query_text, top_k=10)
        # Check chunk content match by comparing content field against corpus.
        returned_content_ids: set[str] = set()
        for r in results:
            # Reconstruct doc id from content by matching corpus.
            for cid, text in _CORPUS:
                if text == r["content"]:
                    returned_content_ids.add(cid)
        for expected_id in expected_ids:
            total_expected += 1
            if expected_id in returned_content_ids:
                total_hit += 1

    recall = total_hit / total_expected if total_expected > 0 else 0.0
    assert recall >= 0.80, (
        f"Recall@10={recall:.2%} below 80% threshold "
        f"({total_hit}/{total_expected} expected chunks found in top-10). "
        f"Check FTS5 Porter stemmer and tokenizer configuration."
    )


def test_fts5_match_returns_nonzero_count(tmp_path: Path) -> None:
    """Verify chunks_fts MATCH works directly after ingest."""
    kb_id = f"fts5-match-{uuid.uuid4().hex[:8]}"
    backend = _build_fresh_backend(tmp_path, kb_id)

    docs = [
        {
            "id": "c-unique",
            "content": "ZXQUNIQUETOKENZXQ for FTS5 match verification",
            "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}"},
        }
    ]
    backend.index(kb_id=kb_id, documents=docs)
    results = backend.search(kb_id=kb_id, text="ZXQUNIQUETOKENZXQ", top_k=5)
    assert len(results) > 0, "FTS5 MATCH returned no results for a known unique token"
