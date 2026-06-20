#!/usr/bin/env python3
"""Standalone recall benchmark for TextvecBackend vs. ChromadbBackend.

Usage::

    python bin/benchmark_recall.py \\
        --kb-id my-kb \\
        --queries-file queries.jsonl \\
        --baseline-backend chromadb \\
        --target-backend textvec \\
        --top-k 10

TODO (Phase C / followup items)
--------------------------------
1. Heavy fixture loading: the 16-transcript curated corpus at
   pipeline_mcp_data/collections/curated/agent-kb-textvec-sources--curated.json
   is not yet wired up in this script because (a) it requires the MCP
   pipeline stack to be running and (b) the KB ingest would take minutes
   for a fair comparison. This script instead ships a synthetic 100-chunk
   corpus generator (``--generate-corpus N``) that is fast and repeatable.
   Wire the heavy fixture in Phase C once kb_migrate textvec is available.

2. ChromadbBackend comparison path: the ``--baseline-backend chromadb``
   path is a stub in this version. Full comparison requires a running
   ChromaDB instance and an embedder. The stub logs a warning and skips
   the baseline run rather than crashing. Wire it fully in Phase C.

3. MRR and mean-rank metrics are computed here using the synthetic
   ground-truth labels from the queries file. For the real fixture,
   generate ground-truth via human annotation or the pipeline's
   existing relevance-judgement tooling.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

_SYNTHETIC_TOPICS = [
    ("neural networks", "Neural networks learn distributed representations from raw data."),
    ("attention mechanism", "Attention mechanisms compute weighted sums over input sequences."),
    ("transformer architecture", "Transformers stack self-attention and feed-forward layers."),
    ("gradient descent", "Gradient descent updates weights by following the loss gradient."),
    (
        "retrieval augmented generation",
        "RAG combines retrieval with generation to ground LLM answers.",
    ),
    (
        "vector embeddings",
        "Dense vector embeddings encode semantic similarity in high-dimensional space.",
    ),
    ("BM25 ranking", "BM25 ranks documents by term frequency and inverse document frequency."),
    ("knowledge graphs", "Knowledge graphs represent entities and relations as nodes and edges."),
    ("fine tuning", "Fine-tuning adapts pretrained models to task-specific datasets."),
    (
        "chunking strategy",
        "Chunking splits documents into smaller pieces for embedding and retrieval.",
    ),
    ("FTS5 sqlite", "SQLite FTS5 provides full-text search with Porter stemming and BM25 ranking."),
    (
        "retrieval pipeline",
        "The retrieval pipeline indexes source documents and retrieves top-k chunks.",
    ),
    (
        "language model pretraining",
        "Pretraining on large corpora builds general-purpose representations.",
    ),
    ("sparse retrieval", "Sparse retrieval uses term-overlap signals like TF-IDF and BM25."),
    (
        "dense retrieval",
        "Dense retrieval uses neural embeddings to find semantically similar text.",
    ),
    (
        "data augmentation",
        "Data augmentation expands training data via paraphrasing and backtranslation.",
    ),
    ("cross encoder reranking", "Cross-encoders rerank candidate documents with deeper attention."),
    ("embedding dimensionality", "Higher embedding dimensions capture more semantic nuances."),
    (
        "context window",
        "The context window limits how many tokens the model can attend to at once.",
    ),
    ("prompt engineering", "Prompt engineering designs inputs that elicit better model responses."),
]


def generate_synthetic_corpus(n: int = 100, kb_id: str = "bench") -> list[dict[str, Any]]:
    """Generate n synthetic chunks drawn cyclically from _SYNTHETIC_TOPICS."""
    docs = []
    for i in range(n):
        topic, base_text = _SYNTHETIC_TOPICS[i % len(_SYNTHETIC_TOPICS)]
        variation = f" Example {i}: {base_text} (variant {i % 5})"
        docs.append(
            {
                "id": f"chunk-{i:04d}",
                "content": base_text + variation,
                "metadata": {
                    "kb_id": kb_id,
                    "source_id": "src-bench",
                    "source_type": "file",
                    "topic": topic,
                },
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Default synthetic queries with ground truth
# ---------------------------------------------------------------------------

_DEFAULT_QUERIES = [
    {
        "query": "attention mechanism transformer",
        "relevant_ids": ["chunk-0001", "chunk-0021", "chunk-0041", "chunk-0061", "chunk-0081"],
    },
    {
        "query": "BM25 retrieval ranking",
        "relevant_ids": ["chunk-0006", "chunk-0026", "chunk-0046", "chunk-0066", "chunk-0086"],
    },
    {
        "query": "neural networks gradient descent",
        "relevant_ids": ["chunk-0000", "chunk-0003", "chunk-0020", "chunk-0023"],
    },
    {
        "query": "vector embeddings dense semantic",
        "relevant_ids": ["chunk-0005", "chunk-0014", "chunk-0025", "chunk-0034"],
    },
    {
        "query": "FTS5 sqlite porter stemming",
        "relevant_ids": ["chunk-0010", "chunk-0030", "chunk-0050", "chunk-0070", "chunk-0090"],
    },
    {
        "query": "retrieval augmented generation RAG",
        "relevant_ids": ["chunk-0004", "chunk-0024", "chunk-0044", "chunk-0064", "chunk-0084"],
    },
    {
        "query": "fine tuning pretrained model",
        "relevant_ids": ["chunk-0008", "chunk-0028", "chunk-0048", "chunk-0068", "chunk-0088"],
    },
    {
        "query": "knowledge graph entities relations",
        "relevant_ids": ["chunk-0007", "chunk-0027", "chunk-0047", "chunk-0067", "chunk-0087"],
    },
    {
        "query": "sparse retrieval TF-IDF",
        "relevant_ids": ["chunk-0013", "chunk-0033", "chunk-0053", "chunk-0073", "chunk-0093"],
    },
    {
        "query": "cross encoder reranking candidates",
        "relevant_ids": ["chunk-0016", "chunk-0036", "chunk-0056", "chunk-0076", "chunk-0096"],
    },
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def recall_at_k(results: list[dict], relevant_ids: list[str], k: int) -> float:
    """Recall@k: fraction of relevant_ids found in top-k results."""
    if not relevant_ids:
        return 0.0
    top_k_ids = {r.get("chunk_id") or r.get("id", "") for r in results[:k]}
    # Also check content field for chunk id via metadata.
    for r in results[:k]:
        meta = r.get("metadata", {})
        chunk_id = meta.get("chunk_id") or ""
        top_k_ids.add(chunk_id)
    hits = sum(1 for rid in relevant_ids if rid in top_k_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(results: list[dict], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank: 1/position of first relevant result."""
    relevant_set = set(relevant_ids)
    for i, r in enumerate(results, start=1):
        chunk_id = r.get("chunk_id") or r.get("id") or r.get("metadata", {}).get("chunk_id", "")
        if chunk_id in relevant_set:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Backend runner
# ---------------------------------------------------------------------------


def run_textvec_search(
    backend,
    kb_id: str,
    query_text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run a textvec search and return results."""
    raw = backend.search(kb_id=kb_id, text=query_text, top_k=top_k)
    # Annotate with doc index for ground-truth matching.
    results = []
    for i, r in enumerate(raw):
        results.append({**r, "rank": i + 1})
    return results


def run_chromadb_search(
    kb_id: str,
    query_text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """ChromaDB search stub — wired in Phase C.

    TODO (Phase C): wire KnowledgebaseService + ChromadbBackend here.
    """
    print(
        f"[WARN] ChromaDB baseline search is a stub in Phase B — skipping query: {query_text!r}",
        file=sys.stderr,
    )
    return []


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_textvec_backend(
    saves_dir: Path,
    kb_id: str,
    corpus: list[dict],
) -> Any:
    """Create a TextvecBackend, create DB, and ingest corpus."""

    from agent_knowledgebase.backends.textvec_backend import TextvecBackend
    from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
    from agent_knowledgebase.database import Database

    settings = Settings(saves_dir=saves_dir)
    safe = sanitize_kb_dir_name(kb_id)
    db_path = settings.kb_db_path(safe)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path=db_path)
    db._conn.execute(
        "INSERT OR IGNORE INTO knowledgebases "
        "(id, name, description, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, kb_id, "benchmark", "{}", "2024-01-01", "2024-01-01"),
    )
    db._conn.execute(
        "INSERT OR IGNORE INTO sources "
        "(id, kb_id, source_type, uri, status) VALUES (?, ?, ?, ?, ?)",
        ("src-bench", kb_id, "file", "/benchmark/corpus.txt", "ingested"),
    )
    db._conn.commit()

    backend = TextvecBackend(settings, service=None)
    print(f"Indexing {len(corpus)} chunks into textvec KB '{kb_id}'...")
    backend.index(kb_id=kb_id, documents=corpus)
    print("Indexing complete.")
    return backend


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(rows: list[dict], cols: list[str]) -> None:
    """Print a simple ASCII table."""
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    separator = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired recall@k benchmark: TextvecBackend vs. ChromadbBackend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--kb-id", default=f"bench-{uuid.uuid4().hex[:8]}", help="KB id to use")
    parser.add_argument(
        "--queries-file", type=Path, default=None, help="JSONL file with queries + relevant_ids"
    )
    parser.add_argument("--baseline-backend", choices=["chromadb", "textvec"], default="chromadb")
    parser.add_argument("--target-backend", choices=["chromadb", "textvec"], default="textvec")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--generate-corpus",
        type=int,
        default=100,
        metavar="N",
        help="Generate N synthetic chunks (default 100)",
    )
    parser.add_argument(
        "--saves-dir",
        type=Path,
        default=None,
        help="Directory for KB storage (default: system tmpdir)",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Write results as JSON to this path"
    )
    args = parser.parse_args()

    # Resolve saves dir.
    import tempfile

    if args.saves_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="kb_bench_")
        saves_dir = Path(tmpdir)
        _owns_tmpdir = True
    else:
        saves_dir = args.saves_dir
        saves_dir.mkdir(parents=True, exist_ok=True)
        _owns_tmpdir = False

    kb_id = args.kb_id

    # Load corpus.
    corpus = generate_synthetic_corpus(n=args.generate_corpus, kb_id=kb_id)
    print(f"Using {len(corpus)}-chunk synthetic corpus (benchmark kb_id={kb_id!r})")
    print("TODO (Phase C): wire pipeline_mcp_data/collections/curated/ fixture here\n")

    # Load queries.
    if args.queries_file and args.queries_file.is_file():
        with args.queries_file.open() as fh:
            queries = [json.loads(line) for line in fh if line.strip()]
        print(f"Loaded {len(queries)} queries from {args.queries_file}")
    else:
        queries = _DEFAULT_QUERIES
        print(f"Using {len(queries)} built-in synthetic queries\n")

    # Setup target backend (textvec).
    target_backend = None
    if args.target_backend == "textvec":
        target_backend = setup_textvec_backend(saves_dir, kb_id, corpus)
    else:
        print("[WARN] Non-textvec target backend is a stub in Phase B.", file=sys.stderr)

    # Run benchmark.
    rows: list[dict[str, Any]] = []
    target_recall_sum = 0.0
    baseline_recall_sum = 0.0
    target_mrr_sum = 0.0
    n_queries = len(queries)

    for qi, q_item in enumerate(queries):
        query_text = q_item.get("query", "")
        relevant_ids = q_item.get("relevant_ids", [])

        # Target backend.
        if target_backend is not None:
            t_results = run_textvec_search(target_backend, kb_id, query_text, args.top_k)
        else:
            t_results = []

        # Baseline backend.
        if args.baseline_backend == "chromadb":
            b_results = run_chromadb_search(kb_id, query_text, args.top_k)
        else:
            b_results = (
                run_textvec_search(target_backend, kb_id, query_text, args.top_k)
                if target_backend
                else []
            )

        t_recall = recall_at_k(t_results, relevant_ids, args.top_k)
        b_recall = recall_at_k(b_results, relevant_ids, args.top_k)
        t_mrr = reciprocal_rank(t_results, relevant_ids)
        target_recall_sum += t_recall
        baseline_recall_sum += b_recall
        target_mrr_sum += t_mrr

        rows.append(
            {
                "query": query_text[:40],
                "target_recall@k": f"{t_recall:.2f}",
                "baseline_recall@k": f"{b_recall:.2f}",
                "target_mrr": f"{t_mrr:.2f}",
                "n_relevant": len(relevant_ids),
            }
        )

    # Summary.
    summary = {
        "kb_id": kb_id,
        "top_k": args.top_k,
        "n_queries": n_queries,
        "target_backend": args.target_backend,
        "baseline_backend": args.baseline_backend,
        "target_mean_recall_at_k": round(target_recall_sum / n_queries, 4) if n_queries else 0.0,
        "baseline_mean_recall_at_k": round(baseline_recall_sum / n_queries, 4)
        if n_queries
        else 0.0,
        "target_mean_mrr": round(target_mrr_sum / n_queries, 4) if n_queries else 0.0,
        "paired_retention_rate": (
            round(target_recall_sum / max(baseline_recall_sum, 1e-9), 4)
            if baseline_recall_sum > 0
            else None
        ),
    }

    print("\n=== Per-Query Results ===")
    print_table(rows, ["query", "target_recall@k", "baseline_recall@k", "target_mrr", "n_relevant"])

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.output_json:
        output = {"summary": summary, "per_query": rows}
        args.output_json.write_text(json.dumps(output, indent=2))
        print(f"\nResults written to {args.output_json}")

    # Cleanup tmpdir if we created it.
    if _owns_tmpdir:
        import shutil

        shutil.rmtree(saves_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
