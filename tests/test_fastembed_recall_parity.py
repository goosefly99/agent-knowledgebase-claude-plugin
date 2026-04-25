"""Phase 5 — fastembed-MiniLM vs sentence-transformers-MiniLM recall parity.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[5].tasks[10] — Jaccard@10 of top-10
        neighbours between fastembed-MiniLM and HF-MiniLM on a fixed
        corpus, assert >= 0.95 (validation finding f-09 sharpening).

What's being tested
-------------------

* Build a fixed ~50-document corpus.
* Embed every doc + a fixed query under both fastembed-MiniLM and
  sentence-transformers-MiniLM.
* Compute the top-10 nearest neighbours (cosine) under each.
* Assert ``Jaccard@10 >= 0.95`` — the int8 quantization should not
  meaningfully shift the neighbour set, which is what justifies
  treating fastembed as a drop-in replacement for the
  similar-vector-geometry sentence-transformers path.

Skipped when either fastembed or sentence-transformers is not
installed. The Phase 5 default install carries NEITHER
(``[embed-local-onnx]`` and ``[embed-local-st]`` are both opt-in).
"""

from __future__ import annotations

import math

import pytest

# Skip the entire module unless BOTH local-embedding stacks are available.
fastembed = pytest.importorskip("fastembed")
sentence_transformers = pytest.importorskip("sentence_transformers")

from agent_knowledgebase.services.embeddings import (  # noqa: E402
    FastembedEmbedder,
    SentenceTransformerEmbedder,
)  # noqa: E402


# A fixed 50-doc corpus. Mix of topical clusters so the neighbour set
# is non-trivial — random short strings would collapse all distances.
_CORPUS = [
    # Cluster: weather
    "It is raining heavily today in the mountains.",
    "Snowfall expected later this evening across the valley.",
    "Sunny skies forecast for the entire weekend ahead.",
    "Heavy fog reduces visibility on the morning commute.",
    "Thunderstorms predicted along the coastal regions tonight.",
    # Cluster: cooking
    "Slowly simmer the tomato sauce for two hours until thick.",
    "Whisk the eggs and sugar together until pale and fluffy.",
    "Roast the vegetables at high heat for caramelization.",
    "Knead the dough until smooth and elastic to the touch.",
    "Marinate the chicken overnight for deeper flavor penetration.",
    # Cluster: programming
    "Refactor the function to reduce its cyclomatic complexity.",
    "The unit tests cover the happy path but miss the edge cases.",
    "Migrate the database schema using the additive ALTER TABLE pattern.",
    "Use a binary search to locate the value in the sorted array.",
    "The compiler emits a deprecation warning for that legacy API.",
    # Cluster: sports
    "The home team won the championship in overtime last night.",
    "The marathon runner trained for six months before the race.",
    "She scored the winning goal in the final minute of play.",
    "The basketball coach drew up a play during the timeout.",
    "The tennis player aced the serve to break match point.",
    # Cluster: animals
    "The cat curled up on the windowsill in the afternoon sun.",
    "Wild dolphins were spotted near the harbor at dawn.",
    "The hawk circled high above searching for its prey.",
    "Bees pollinate the wildflowers throughout the meadow.",
    "Migrating geese flew south in their characteristic V formation.",
    # Cluster: music
    "The orchestra played a Mozart symphony in the grand hall.",
    "The guitarist tuned his instrument before stepping on stage.",
    "Jazz improvisation demands deep familiarity with chord changes.",
    "The choir rehearsed the hymn until the harmonies blended perfectly.",
    "She composed the song on a quiet rainy Sunday afternoon.",
    # Cluster: travel
    "The overnight train winds through three countries by morning.",
    "Pack lightly for the trek across the mountain passes.",
    "Flights to coastal cities tend to spike during summer holidays.",
    "The hostel by the beach offers cheap rooms and a friendly bar.",
    "Local cuisine gives the clearest window into a foreign culture.",
    # Cluster: science
    "The experiment confirmed the predicted reaction kinetics.",
    "Quantum entanglement defies our classical intuitions about locality.",
    "Climate models project warmer winters in the northern hemisphere.",
    "The telescope captured an image of the distant nebula.",
    "Genome sequencing has dropped to a few hundred dollars per sample.",
    # Cluster: gardening
    "Mulch around the tomato plants to retain soil moisture.",
    "Prune the rose bushes after the first hard frost.",
    "Compost coffee grounds and eggshells into the vegetable beds.",
    "The herbs thrive in a sunny windowsill near the kitchen.",
    "Plant the bulbs in autumn for a spring blooming display.",
    # Cluster: history
    "The treaty ended a century of intermittent border conflicts.",
    "Archaeologists unearthed pottery dating to the Bronze Age.",
    "The empire collapsed under economic strain and external pressure.",
    "The revolution sparked a wave of constitutional reforms.",
    "Medieval scribes copied texts by hand in monastery scriptoria.",
]
assert len(_CORPUS) == 50, f"corpus must be 50 docs, got {len(_CORPUS)}"

_QUERY = "Refactor the helper function to handle the new edge cases cleanly."


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (assumes non-zero magnitudes)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _top_k_indices(query_vec: list[float], doc_vecs: list[list[float]], k: int) -> set[int]:
    """Return indices of the top-k highest-cosine docs for the query."""
    scored = [(i, _cosine(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return {i for i, _ in scored[:k]}


def _build_pair() -> tuple[FastembedEmbedder, SentenceTransformerEmbedder]:
    """Return a (fastembed, sentence-transformers) embedder pair, both
    targeting MiniLM-class 384-dim models. Skip if either fails to
    construct (model file unavailable in offline test env, etc.)."""
    try:
        fe = FastembedEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        try:
            fe = FastembedEmbedder(model_name="BAAI/bge-small-en-v1.5")
        except Exception as exc2:  # noqa: BLE001
            pytest.skip(
                f"No 384-dim fastembed model available locally: {exc!r} / {exc2!r}"
            )
    try:
        st = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"sentence-transformers MiniLM unavailable: {exc!r}")
    # Dimension parity: both must agree on 384, otherwise the parity
    # comparison is meaningless.
    if fe.probe_dimension() != st.dimension:
        pytest.skip(
            f"dimension mismatch: fastembed={fe.probe_dimension()} st={st.dimension}"
        )
    return fe, st


def test_fastembed_minilm_jaccard_at_10_matches_st_minilm() -> None:
    """Jaccard@10 of fastembed-MiniLM vs sentence-transformers-MiniLM on a
    fixed 50-doc corpus must be >= 0.95 — proving int8 quantization
    does not meaningfully shift the top-10 neighbour set."""
    fe, st = _build_pair()

    fe_docs = fe.embed(_CORPUS)
    st_docs = st.embed(_CORPUS)
    fe_query = fe.embed_query(_QUERY)
    st_query = st.embed_query(_QUERY)

    fe_top10 = _top_k_indices(fe_query, fe_docs, k=10)
    st_top10 = _top_k_indices(st_query, st_docs, k=10)

    intersection = len(fe_top10 & st_top10)
    union = len(fe_top10 | st_top10)
    jaccard = intersection / union if union else 1.0

    assert jaccard >= 0.95, (
        f"fastembed/sentence-transformers MiniLM Jaccard@10 = {jaccard:.3f} "
        f"(need >= 0.95). fe_top10={sorted(fe_top10)} st_top10={sorted(st_top10)}"
    )
