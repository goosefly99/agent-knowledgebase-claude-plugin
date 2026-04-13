"""Integration: new config values flow into the actual services."""
from __future__ import annotations

from pathlib import Path

from agent_knowledgebase.config import Settings


class TestChunkTokenEncoding:
    def test_chunk_config_uses_setting(self, tmp_path: Path) -> None:
        from agent_knowledgebase.ingestors import ChunkConfig
        cfg = Settings(saves_dir=tmp_path, chunk_token_encoding="o200k_base")
        cc = ChunkConfig.from_settings(cfg)
        assert cc.token_encoding == "o200k_base"


class TestIngestExcludedDirs:
    def test_directory_ingestor_uses_setting(self, tmp_path: Path) -> None:
        from agent_knowledgebase.ingestors.directory import excluded_dirs_for
        cfg = Settings(saves_dir=tmp_path, ingest_excluded_dirs=["only_this"])
        assert excluded_dirs_for(cfg) == {"only_this"}


class TestHybridWeightsInQuery:
    def test_query_orchestrator_uses_settings(self, tmp_path: Path) -> None:
        from agent_knowledgebase.services.query import hybrid_weights_from
        cfg = Settings(
            saves_dir=tmp_path,
            query_hybrid_vector_weight=0.55,
            query_hybrid_fts_weight=0.45,
            query_hybrid_fetch_multiplier=3,
        )
        vw, fw, mult = hybrid_weights_from(cfg)
        assert vw == 0.55
        assert fw == 0.45
        assert mult == 3


class TestDefaultTopK:
    def test_default_top_k_from_settings(self, tmp_path: Path) -> None:
        from agent_knowledgebase.services.query import default_top_k
        cfg = Settings(saves_dir=tmp_path, query_default_top_k=15)
        assert default_top_k(cfg) == 15
