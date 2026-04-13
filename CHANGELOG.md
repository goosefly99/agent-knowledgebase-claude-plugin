# Changelog

## 0.2.0 — 2026-04-12

### Added
- User-level (`~/.agent-kb/config.json`) and project-level (`./.agent-kb/config.json`) JSON config files, layered under environment variables.
- Six new config knobs: `chunk.token_encoding`, `query.default_top_k`, `query.hybrid.vector_weight`, `query.hybrid.fts_weight`, `query.hybrid.fetch_multiplier`, `ingest.excluded_dirs`.
- Five new MCP tools: `kb_config_path`, `kb_config_show`, `kb_config_get`, `kb_config_set`, `kb_config_validate`.
- Hybrid-weights-sum-to-1.0 validator on `Settings`.
- `ingest.excluded_dirs` list is consolidated into `Settings` as the single source of truth (previously duplicated in `DirectoryIngestor`).

### Changed
- `Settings` now layers defaults → user JSON → project JSON → env (highest priority).
- `token_chunk` and `DirectoryIngestor` now read their behavior from `Settings` instead of hardcoded module-level constants.
- `KnowledgebaseService.query/search/hybrid_query` and MCP `kb_query/kb_search` tools default `top_k` to `Settings.query_default_top_k` (via `None` sentinel) instead of the hardcoded `10`.

### Preserved
- `AGENT_KB_SAVES_DIR` remains required and env-only.
- Secrets (`AGENT_KB_OPENAI_API_KEY`, `AGENT_KB_PINECONE_API_KEY`) remain env-only. JSON files reject them.

## 0.1.0 — 2026-04-11

Initial release.
