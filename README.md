# Ember MCP — Persistent Memory with Drift Detection to Prevent Hallucination

**Local-first memory server for LLMs that uses Voronoi partitioning to manage context and prevent hallucinations caused by stale data.** Stop re-explaining your stack every time you open a new chat window — Ember gives your AI a permanent memory that follows you from Claude to Cursor, automatically discarding outdated decisions so you never get code based on the architecture you abandoned last month.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

Ember MCP is a Model Context Protocol server that provides LLMs with long-term memory without cloud dependencies. Unlike standard vector stores that simply retrieve the most similar chunks, Ember actively manages knowledge density and freshness to prevent the retrieval of obsolete information. If you built an authentication system using JWTs six months ago but migrated to OAuth last week, Ember ensures the AI suggests OAuth patterns, not the deprecated JWT code. When you dump months of meeting transcripts into the chat, it distinguishes between the feature requirements you set in January and the pivot you decided on in March. Best of all, this context follows you everywhere: debug a backend issue in Claude Desktop during the day, and continue refactoring the exact same context in Cursor at night.

## Key Features

- **Cross-Session:** Memories persist across conversations and different MCP clients. Close your laptop on Friday, open a completely fresh chat session on Monday, and the AI picks up exactly where you left off without needing a summary.
- **100% Privacy:** Runs locally on CPU (~500MB disk, ~200MB RAM). No API keys or cloud vector DBs required. Paste client NDAs, proprietary algorithms, or financial records without worry — your data never leaves your machine or touches a cloud vector store.
- **Temporal Intelligence:** Memories are ranked by recency (exponential decay) and access frequency. Stale data is explicitly penalized. Tell the AI you're using React 17 today, and when you upgrade to React 19 next month, it understands the old syntax is history and shouldn't be referenced.
- **Drift Detection:** Automatically detects when knowledge regions shift using statistical monitoring, flagging outdated memories without manual intervention. As you slowly migrate a legacy service from REST to GraphQL over several weeks, the system notices the pattern shift and stops suggesting REST endpoints.
- **Source Linking:** Memories trace back to their origin files, allowing deep recall of source context. When the AI claims "we decided to use Kubernetes," it doesn't just hallucinate a memory — it points you to the specific meeting note or architecture doc where that decision was recorded.
- **Zero-Config:** Works automatically via embedded server instructions. No vector databases to spin up, no embeddings to configure, and no "memory dashboard" to manually prune — it just runs in the background.

## Why Drift Detection Matters

Standard vector databases suffer from "semantic collision" when a project evolves. Old, obsolete memories often have high semantic similarity to new queries, causing the LLM to hallucinate based on past states. Ember solves this by detecting distributional shifts in the vector space.

### What This Means for You

You've been working on a project for months, and inevitably, requirements have changed and tech stacks have evolved. Without drift detection, an AI sees your entire history as equally valid, confidently suggesting the library you abandoned two months ago. Ember detects when a topic cluster has shifted — like a migration from Redux to Zustand — and automatically treats the old data as stale. You get answers based on your current architecture, not the ghosts of your previous decisions, without ever having to manually "clean up" the AI's memory.

### The Scenario

**January:** You tell Claude your project uses **PostgreSQL**. Ember stores memories about schemas and drivers in the "databases" region of the vector space.

**April:** You migrate to **MongoDB**. You store new memories about documents and collections.

### Without Ember (Standard Vector Store)

The old "we use PostgreSQL" memory remains semantically similar to questions about "database queries."

**Result:** Claude confidently provides PostgreSQL SQL syntax, hallucinating based on stale memory despite your migration.

### With Ember Drift Detection

1. **Detection:** Ember notices the "databases" Voronoi cell has shifted significantly. The mean vector has moved and density has changed due to the influx of MongoDB concepts.
2. **Flagging:** The system identifies the statistical drift and automatically flags the older PostgreSQL memories as "stale."
3. **Result:** When you ask about databases, the stale PostgreSQL memory is penalized (ranked 10x lower). Claude retrieves only the active MongoDB context.

## Get Started

→ **[ember.timolabs.dev](https://ember.timolabs.dev/)** — installation, docs, and free trial

The installer automatically:
- Detects which MCP clients you have installed (Claude Desktop, Claude Code, Cursor, Windsurf)
- Registers Ember with each one — no JSON editing required
- Creates the local storage directory
- Downloads the embedding model

Restart your AI client and you're ready to go. Your AI now has persistent memory.

### Verify

```bash
ember-mcp status
```

Shows which clients are registered and how many memories are stored.

### Manual Configuration

If you use an MCP client that isn't auto-detected, add this to its config:

```json
{
  "mcpServers": {
    "ember": {
      "command": "ember-mcp",
      "args": ["run"]
    }
  }
}
```

## How It Works

Ember combines classical statistics with vector search to manage knowledge density and freshness:

### What This Means for You

You don't need to learn a new syntax or manage a "knowledge base." You just talk to your AI as you normally would. Behind the scenes, Ember clusters your conversations by topic, tracks which information is fresh versus stale, and injects only the relevant context into your current session.

1. **Local Embeddings:** Uses `all-MiniLM-L6-v2` to generate 384-dimensional vectors locally on CPU.
2. **Voronoi Partitioning:** The vector space is divided into stable regions using 16 frozen centroids (L2-normalized, seed=42). This clusters knowledge naturally (e.g., coding rules in one cell, business logic in another).
3. **FAISS Search:** Meta's FAISS library handles high-speed similarity search with custom ID mapping.
4. **Welford Statistics:** Tracks streaming mean and variance per cell to monitor knowledge density over time.
5. **Temporal Intelligence:** Applies decay scoring (exponential half-lives based on importance) and boosts ranking for frequently accessed items.
6. **Drift Detection Pipeline:** A multi-stage statistical pipeline to auto-flag staleness:
   - Monitors per-cell metrics (mass delta, mean shift, covariance change).
   - Constructs a k-NN adjacency graph between centroids.
   - Applies Laplacian smoothing to suppress isolated noise.
   - Uses adaptive thresholding to identify significant shifts in knowledge regions.

## Tools

Ember exposes **11 tools** to the LLM:

| Tool | Description |
|------|-------------|
| `ember_store` | Save a named memory with importance level and optional tags |
| `ember_recall` | Semantic search with temporal scoring across all memories |
| `ember_deep_recall` | Recall + automatically read source files behind the embers |
| `ember_learn` | Auto-capture key information from conversation (facts, preferences, decisions) |
| `ember_contradict` | Mark outdated memory stale and store corrected version |
| `ember_list` | List all stored memories, optionally filtered by tag |
| `ember_delete` | Remove a memory by ID |
| `ember_inspect` | View Voronoi cell distribution, statistics, and density |
| `ember_auto` | Auto-retrieve relevant context at conversation start with temporal ranking |
| `ember_save_session` | Save session summary, decisions, and next steps with source linking |
| `ember_drift_check` | Run drift detection — flag stale memories in shifting knowledge regions |

## Prompts

| Prompt | Description |
|--------|-------------|
| `start_session` | Load memory at conversation start |
| `end_session` | Save context before ending |
| `remember` | Store something user wants persisted |

## Storage

All data is stored locally in `~/.ember/`.

```
~/.ember/
├── config.json
├── embers/*.json
├── index/
│   ├── vectors.faiss
│   └── id_map.json
└── cells/
    ├── centroids.npy
    └── stats.db        # SQLite: cell_stats + cell_stats_snapshot
```

To reset: `rm -rf ~/.ember`

To backup: copy the `~/.ember` directory.

## Requirements

- **Python:** 3.10+
- **Disk:** ~500MB (embedding model + dependencies)
- **RAM:** ~200MB overhead
- **OS:** macOS, Linux, Windows (WSL)

## License

MIT — Built by [Arkya AI](https://github.com/ArkyaAI)
