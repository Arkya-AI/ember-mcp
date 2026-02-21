# Changelog

All notable changes to Ember MCP are documented here.

## [0.2.0] — 2026-02-21

### New Tools
- **`ember_compact`** — AI-powered memory compaction. Analyze mode identifies stale/shadowed candidates; execute mode replaces content with an LLM-generated summary and re-embeds the vector.
- **`ember_actionable`** — List embers with active task status (`open` or `in_progress`), sorted by priority.
- **`ember_set_status`** — Update the task status of any ember (`open`, `in_progress`, `done`, or clear).

### Enhancements
- **`ember_store`** — New `status` parameter for task tracking (`open`, `in_progress`, `done`). New `edges` parameter for typed knowledge graph edges (`depends_on`, `child_of`, `context_for`). Invalid `importance` or `status` values now return an error string instead of silently coercing to defaults.
- **`ember_graph_search`** — New `edge_types` parameter to filter BFS traversal by edge type.
- **`ember_auto`** — Session continuity boost: session-sourced embers receive a 1.5× HESTIA score multiplier; the most recent session ember receives 2.0×. Candidate pool expanded from 5 to 10 before boosting.

### Models
- `Ember` model: new fields `status`, `is_compacted`, `original_content_length` (all optional with defaults — no migration required for existing embers).
- New constants: `VALID_EDGE_TYPES`, `USER_EDGE_TYPES`.

### Security
- **Path traversal fix (C1):** `ember_deep_recall` now restricts `source_path` reads to within `Path.home()`. Paths outside this boundary are skipped with a `[Skipped: path outside home directory]` annotation.

### Registry
- Added `server.json` for MCP Registry listing at `registry.modelcontextprotocol.io`.

## [0.1.0] — Initial release

- 14 MCP tools: `ember_store`, `ember_recall`, `ember_read`, `ember_deep_recall`, `ember_learn`, `ember_contradict`, `ember_list`, `ember_delete`, `ember_inspect`, `ember_auto`, `ember_save_session`, `ember_drift_check`, `ember_health`, `ember_recompute_shadows`, `ember_explain`.
- FAISS-based semantic search with 16-cell Voronoi partitioning.
- Shadow-Decay framework with HESTIA scoring.
- SQLite knowledge graph with `related`, `shadow`, and `supersedes` edge types.
- Local-only storage at `~/.ember/` — no cloud dependencies.
