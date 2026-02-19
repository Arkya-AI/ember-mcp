import asyncio
import os
import signal
from datetime import datetime, timezone
from typing import Optional, List

import aiofiles
import numpy as np
from mcp.server.fastmcp import FastMCP

from ember.core import VectorEngine
from ember.storage import StorageManager
from ember.config import load_config
from ember.models import Ember, DECAY_HALF_LIVES
from ember.utils import (
    compute_temporal_score,
    compute_shadow_potential,
    compute_shadow_load,
    compute_topic_vitality,
    compute_hestia_score,
    compute_hallucination_risk,
    detect_kg_edges,
    update_region_stats,
)

# ---------------------------------------------------------------------------
# Token Budget Constants
# ---------------------------------------------------------------------------
PREVIEW_CHARS = 150          # Max chars of ember preview in recall/auto
SOURCE_FILE_MAX_CHARS = 2000 # Max chars per source file in deep recall
MAX_SOURCE_FILES = 3         # Max source files to read in deep recall

# Initialize FastMCP with Shadow-Decay instructions
mcp = FastMCP(
    "Ember",
    instructions=(
        "Ember is persistent memory. On conversation start, call ember_auto with a "
        "summary of the user's first message. After each response where the user "
        "shared facts, preferences, decisions, or corrections, call ember_learn "
        "silently. Never announce memory operations. Never mention internal mechanics "
        "(vectors, scores, cells). Use recalled memories naturally. When correcting "
        "old info, use ember_contradict. For source file detail, use ember_deep_recall. "
        "Use ember_read to get full content of a specific memory."
    ),
)

# Global singleton instances (lazy loaded)
_engine: Optional[VectorEngine] = None
_storage: Optional[StorageManager] = None


async def _ensure_init():
    """Lazy initialization of the VectorEngine and StorageManager."""
    global _engine, _storage
    if _engine is None or _storage is None:
        config = load_config()
        _engine = VectorEngine(config)
        _storage = StorageManager(config)
        await _storage.init_db()


async def _reload_from_disk():
    """Reload FAISS index and ID mappings from disk to pick up external writes."""
    _engine.reload_index()
    await _storage.reload_id_map()


def _shutdown_handler(signum, frame):
    """Flush dirty index and release locks on SIGTERM/SIGINT."""
    if _engine is not None:
        _engine._save_index_sync()


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


# ---------------------------------------------------------------------------
# Helpers: Preview + Fetch-and-Rerank + Shadow-on-Insert
# ---------------------------------------------------------------------------


def _make_preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    """Create a preview of text for token-efficient search results."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


async def _fetch_and_rerank(
    query_text: str, top_k: int = 5, fetch_multiplier: int = 10
) -> list:
    """
    Fetch-and-Rerank pipeline using HESTIA scoring.

    1. Embed query
    2. Broad FAISS search: k = top_k * fetch_multiplier candidates
    3. Compute topic vitality V(q,t) via radius search
    4. For each candidate: compute HESTIA score using cached shadow_load
    5. Sort by HESTIA score, return top_k as list of (ember, score, breakdown)
    """
    now = datetime.now(timezone.utc)
    config = load_config()

    vector = await _engine.embed(query_text)
    total = _engine.memory_index.ntotal
    fetch_k = min(top_k * fetch_multiplier, total) if total > 0 else 0
    if fetch_k == 0:
        return []

    results = _engine.search(vector, top_k=fetch_k)
    if not results:
        return []

    # Topic vitality: search within radius, get neighbor times
    radius_l2 = VectorEngine.cosine_to_l2(1.0 - config.topic_radius)
    radius_results = _engine.search_radius(vector, radius_l2)

    neighbor_times = []
    neighbor_dists = []
    for faiss_id, dist_sq in radius_results:
        uuid = _storage.int_to_uuid.get(faiss_id)
        if uuid:
            ember = await _storage.get_ember(uuid)
            if ember:
                neighbor_times.append(ember.created_at)
                neighbor_dists.append(dist_sq)

    vitality = compute_topic_vitality(
        neighbor_dists, neighbor_times, now, radius_l2, config.vitality_lambda
    )

    # Score each candidate with HESTIA
    scored = []
    v_max = max(vitality, 0.001)  # prevent division by zero

    for faiss_id, dist in results:
        uuid = _storage.int_to_uuid.get(faiss_id)
        if not uuid:
            continue
        ember = await _storage.get_ember(uuid)
        if not ember:
            continue

        cos_sim = VectorEngine.l2_to_cosine(dist)
        score, breakdown = compute_hestia_score(
            cos_sim,
            ember.shadow_load,
            vitality,
            v_max,
            config.shadow_gamma,
            config.nostalgia_alpha,
        )

        # Update access stats
        ember.last_accessed_at = now
        ember.access_count += 1
        await _storage.update_ember(ember)

        scored.append((ember, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


async def _shadow_on_insert(new_ember: Ember, vector: np.ndarray) -> None:
    """Shadow-on-Insert: update shadow_load on existing neighbors bidirectionally."""
    config = load_config()

    # Ensure vector is 2D (1, dim) for FAISS compatibility
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    results = _engine.search(vector, top_k=config.shadow_k)
    if not results:
        return

    cos_sims = []
    shadow_potentials = []
    neighbor_ids = []

    for faiss_id, dist in results:
        uuid = _storage.int_to_uuid.get(faiss_id)
        if not uuid or uuid == new_ember.ember_id:
            continue

        neighbor = await _storage.get_ember(uuid)
        if not neighbor:
            continue

        cos_sim = VectorEngine.l2_to_cosine(dist)
        phi = compute_shadow_potential(
            cos_sim,
            neighbor.created_at,
            new_ember.created_at,
            config.shadow_delta,
            config.shadow_epsilon,
        )

        cos_sims.append(cos_sim)
        shadow_potentials.append(phi)
        neighbor_ids.append(uuid)

        # Update shadow_load on older neighbors if new ember shadows them harder
        if phi > neighbor.shadow_load:
            neighbor.shadow_load = phi
            neighbor.shadowed_by = new_ember.ember_id
            neighbor.shadow_updated_at = datetime.now(timezone.utc)
            await _storage.update_ember(neighbor)
            await _storage.save_edge(
                neighbor.ember_id, new_ember.ember_id, "shadow", phi
            )

        # Update region stats with conflict density
        region = await _storage.get_region_stats(neighbor.cell_id)
        updated_region = update_region_stats(neighbor.cell_id, phi, region)
        await _storage.update_region(
            updated_region.cell_id,
            updated_region.vitality_score,
            updated_region.shadow_accum,
        )

    # Detect KG edges (related but not shadowing)
    kg_edges = detect_kg_edges(cos_sims, shadow_potentials, neighbor_ids)
    if kg_edges:
        new_ember.related_ids = kg_edges[:5]
        await _storage.update_ember(new_ember)
        for related_id in kg_edges:
            await _storage.save_edge(new_ember.ember_id, related_id, "related", 0.0)


# ---------------------------------------------------------------------------
# MCP Tools: Core Memory Operations
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def ember_store(
    name: str,
    content: str,
    tags: str = "",
    importance: str = "context",
    source_path: str = "",
) -> str:
    """
    Store a memory ember with importance level. The content will be embedded and assigned
    to a Voronoi cell for persistent retrieval.

    Args:
        name: Short descriptive name for this memory
        content: The actual content to remember
        tags: Comma-separated tags for categorization
        importance: One of: fact, decision, preference, context, learning
        source_path: Optional path to the source file this info came from.
            Enables deep recall to read the full source for richer context.
    """
    await _ensure_init()

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    valid_importance = ["fact", "decision", "preference", "context", "learning"]
    if importance not in valid_importance:
        importance = "context"

    vector = await _engine.embed(content)
    cell_id = _engine.assign_cell(vector)

    ember = Ember(
        name=name,
        content=content,
        tags=tag_list,
        cell_id=cell_id,
        importance=importance,
        source="manual",
        source_path=source_path if source_path else None,
    )

    faiss_id = await _storage.save_ember(ember)
    await _engine.add_vector(faiss_id, vector)

    await _shadow_on_insert(ember, vector.flatten())

    half_life = DECAY_HALF_LIVES.get(importance, 30.0)
    return (
        f"Stored ember '{name}' (ID: {ember.ember_id}) in Cell {cell_id}. "
        f"Importance: {importance} (half-life: {int(half_life)}d)"
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_recall(query: str, top_k: int = 5) -> str:
    """
    Retrieve memory embers semantically similar to the query, ranked by HESTIA score.
    Newer, non-shadowed, actively-discussed memories rank higher.
    """
    await _ensure_init()
    await _reload_from_disk()
    now = datetime.now(timezone.utc)

    scored = await _fetch_and_rerank(query, top_k=top_k)

    if not scored:
        return "No embers found."

    lines = []
    for ember, score, breakdown in scored:
        age_days = (now - ember.created_at).total_seconds() / 86400.0
        freshness = "fresh" if age_days < 7 else f"{int(age_days)}d ago"
        stale_mark = " [STALE]" if ember.is_stale else ""
        shadow_mark = f" [shadow:{ember.shadow_load:.1f}]" if ember.shadow_load > 0.1 else ""
        source_note = f"\n  source: {ember.source_path}" if ember.source_path else ""

        preview = _make_preview(ember.content)
        lines.append(
            f"🔥 {ember.name} [id: {ember.ember_id}] "
            f"(score: {score:.2f}, {freshness}{stale_mark}{shadow_mark})\n"
            f"  {preview}{source_note}"
        )

    lines.append("\n→ Use ember_read(id) for full content of any memory.")
    return "\n\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_deep_recall(query: str, top_k: int = 3) -> str:
    """
    Retrieve memory embers AND read their source files for full context.
    Use this when you need deeper detail than what the ember summary provides.

    Flow: semantic search → find embers → read source files on disk → return combined context.

    Args:
        query: What to search for in memory
        top_k: Number of embers to retrieve (default 3, reads source files for each)
    """
    await _ensure_init()
    await _reload_from_disk()
    now = datetime.now(timezone.utc)

    scored = await _fetch_and_rerank(query, top_k=top_k)

    if not scored:
        return "No embers found."

    lines = []
    source_contents = {}  # Deduplicate source file reads

    for ember, score, breakdown in scored:
        age_days = (now - ember.created_at).total_seconds() / 86400.0
        freshness = "fresh" if age_days < 7 else f"{int(age_days)}d ago"
        stale_mark = " [STALE]" if ember.is_stale else ""

        # Deep recall returns full content (this IS the "read the full page" action)
        lines.append(
            f"🔥 {ember.name} [id: {ember.ember_id}] "
            f"(score: {score:.2f}, {freshness}{stale_mark})\n{ember.content}"
        )

        # Read source file if available, not already read, and under cap
        if (
            ember.source_path
            and ember.source_path not in source_contents
            and len(source_contents) < MAX_SOURCE_FILES
        ):
            if os.path.isfile(ember.source_path):
                try:
                    async with aiofiles.open(ember.source_path, mode="r") as f:
                        raw = await f.read()
                    source_contents[ember.source_path] = _make_preview(raw, SOURCE_FILE_MAX_CHARS)
                except Exception as e:
                    source_contents[ember.source_path] = f"[Error reading: {e}]"
            else:
                source_contents[ember.source_path] = "[File not found]"

    output = "\n\n---\n\n".join(lines)

    if source_contents:
        output += "\n\n===== SOURCE FILES =====\n"
        for path, content in source_contents.items():
            output += f"\n--- {path} ---\n{content}\n"

    return output


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def ember_learn(conversation_context: str, source_path: str = "") -> str:
    """
    Auto-capture key information from conversation. Extracts facts, preferences,
    decisions, and learnings — then stores them as temporal embers.

    Call this silently after every substantive user message.
    The LLM should extract and classify the information before calling this.

    Args:
        conversation_context: The key information to capture, formatted as:
            "TYPE: description" where TYPE is fact/decision/preference/learning
            Example: "preference: User prefers TypeScript with strict mode"
        source_path: Optional path to the source file (handoff, document) this info came from.
            When provided, enables deep recall to read the full source for richer context.
    """
    await _ensure_init()
    await _reload_from_disk()

    # Parse the type prefix
    importance = "context"
    content = conversation_context

    for itype in ["fact", "decision", "preference", "learning", "context"]:
        if conversation_context.lower().startswith(f"{itype}:"):
            importance = itype
            content = conversation_context[len(itype) + 1 :].strip()
            break

    # Generate a concise name from first ~60 chars
    name = content[:60].strip()
    if len(content) > 60:
        name = name.rsplit(" ", 1)[0] + "..."

    vector = await _engine.embed(content)
    cell_id = _engine.assign_cell(vector)

    # Check for near-duplicates
    existing = _engine.search(vector, top_k=3)
    for faiss_id, dist in existing:
        if dist < 0.1:  # Very similar (normalized L2 < 0.1 means almost identical)
            uuid = _storage.int_to_uuid.get(faiss_id)
            if uuid:
                existing_ember = await _storage.get_ember(uuid)
                if existing_ember and not existing_ember.is_stale:
                    existing_ember.last_accessed_at = datetime.now(timezone.utc)
                    existing_ember.access_count += 1
                    await _storage.update_ember(existing_ember)
                    return f"Reinforced existing ember: '{existing_ember.name}'"

    ember = Ember(
        name=name,
        content=content,
        tags=["auto-captured", importance],
        cell_id=cell_id,
        importance=importance,
        source="auto",
        source_path=source_path if source_path else None,
    )

    faiss_id = await _storage.save_ember(ember)
    await _engine.add_vector(faiss_id, vector)

    await _shadow_on_insert(ember, vector.flatten())

    return f"Captured {importance}: '{name}'"


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False})
async def ember_contradict(ember_id: str, new_content: str, reason: str = "") -> str:
    """
    Mark an existing memory as stale and store an updated version.
    Use when the user corrects or updates previously stored information.

    Args:
        ember_id: The ID of the ember to mark stale (use ember_recall to find it)
        new_content: The corrected/updated information
        reason: Why the old information is stale
    """
    await _ensure_init()

    # Mark old ember as fully shadowed
    old_ember = await _storage.get_ember(ember_id)
    if not old_ember:
        return f"Ember {ember_id} not found."

    old_ember.is_stale = True
    old_ember.stale_reason = reason or "Superseded by newer information"
    old_ember.shadow_load = 1.0

    # Store new version with supersedes link
    vector = await _engine.embed(new_content)
    cell_id = _engine.assign_cell(vector)

    new_ember = Ember(
        name=old_ember.name,
        content=new_content,
        tags=old_ember.tags,
        cell_id=cell_id,
        importance=old_ember.importance,
        supersedes_id=ember_id,
        source="manual",
    )

    # Set bidirectional links
    old_ember.shadowed_by = new_ember.ember_id
    old_ember.shadow_updated_at = datetime.now(timezone.utc)
    new_ember.superseded_by_id = None  # new ember is the current version

    await _storage.update_ember(old_ember)
    faiss_id = await _storage.save_ember(new_ember)
    await _engine.add_vector(faiss_id, vector)

    # Create supersedes edge
    await _storage.save_edge(ember_id, new_ember.ember_id, "supersedes", 1.0)

    # Shadow-on-Insert for the new ember
    await _shadow_on_insert(new_ember, vector.flatten())

    return (
        f"Updated memory: '{old_ember.name}'. "
        f"Old version fully shadowed. New version: {new_ember.ember_id}"
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_list(tag: str = "", limit: int = 20, offset: int = 0) -> str:
    """List stored memory embers with pagination. Returns metadata only (no content).

    Args:
        tag: Optional tag filter
        limit: Max results per page (default 20)
        offset: Skip this many results (default 0)
    """
    await _ensure_init()
    await _reload_from_disk()

    all_embers = await _storage.list_embers(tag=tag if tag else None)

    if not all_embers:
        return "No embers stored." if not tag else f"No embers with tag '{tag}'."

    total = len(all_embers)
    page = all_embers[offset : offset + limit]

    lines = []
    now = datetime.now(timezone.utc)
    for a in page:
        age_days = (now - a.created_at).total_seconds() / 86400.0
        freshness = "today" if age_days < 1 else f"{int(age_days)}d ago"
        stale = " [STALE]" if a.is_stale else ""
        shadow = f" [shadow:{a.shadow_load:.1f}]" if a.shadow_load > 0.1 else ""
        lines.append(
            f"• {a.name} ({a.importance}) [{freshness}]{stale}{shadow} "
            f"[id: {a.ember_id}]"
        )

    start = offset + 1
    end = min(offset + limit, total)
    header = f"Showing {start}-{end} of {total} embers"
    if end < total:
        header += f" (use offset={end} for more)"
    header += ":"

    return header + "\n" + "\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False})
async def ember_delete(ember_id: str) -> str:
    """Delete a memory ember by its ID (UUID)."""
    await _ensure_init()

    faiss_id = await _storage.delete_ember(ember_id)
    if faiss_id is None:
        return f"Ember {ember_id} not found."

    try:
        await _engine.remove_vector(faiss_id)
    except Exception:
        pass

    return f"Ember {ember_id} deleted."


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_inspect(cell_id: int = -1) -> str:
    """Inspect Voronoi cell health. Shows ember distribution and conflict density."""
    await _ensure_init()
    config = load_config()

    if cell_id >= 0:
        stats = await _storage.get_region_stats(cell_id)
        if not stats:
            return f"Cell {cell_id}: no data"
        return (
            f"Cell {cell_id}: vitality={stats.vitality_score:.3f}, "
            f"conflict_density={stats.shadow_accum:.3f}, "
            f"last_updated={stats.last_updated}"
        )

    # Overview: count embers per cell without loading content
    all_embers = await _storage.list_embers()
    cell_counts = {}
    for e in all_embers:
        cell_counts[e.cell_id] = cell_counts.get(e.cell_id, 0) + 1

    total = len(all_embers)
    lines = [f"Voronoi Cell Map ({config.k_cells} cells, {total} embers):"]
    for i in range(config.k_cells):
        count = cell_counts.get(i, 0)
        bar = "█" * min(count, 20)
        stats = await _storage.get_region_stats(i)
        conflict = f" conflict:{stats.shadow_accum:.2f}" if stats else ""
        lines.append(f"  Cell {i:2d}: {bar} {count}{conflict}")

    return "\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_auto(conversation_context: str) -> str:
    """
    Automatically retrieve relevant memory embers based on conversation context.
    Uses HESTIA scoring — non-shadowed, actively-discussed memories rank higher.
    Call at the start of every conversation.
    """
    await _ensure_init()
    await _reload_from_disk()

    scored = await _fetch_and_rerank(conversation_context, top_k=5)

    if not scored:
        return ""

    lines = []
    has_sources = False
    for ember, score, breakdown in scored:
        stale_note = " (outdated)" if ember.is_stale else ""
        preview = _make_preview(ember.content)
        lines.append(
            f"🔥 {ember.name} [id: {ember.ember_id}]{stale_note}: {preview}"
        )
        if ember.source_path:
            has_sources = True

    if has_sources:
        lines.append("\n→ Use ember_deep_recall for source file context, ember_read(id) for full content.")

    return "\n\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_read(ember_id: str) -> str:
    """
    Read the full content of a specific ember by ID.
    Use this after ember_recall/ember_auto returns previews
    and you need the complete content for a specific memory.

    Args:
        ember_id: The ID of the ember to read (shown in search results as [id: ...])
    """
    await _ensure_init()
    ember = await _storage.get_ember(ember_id)
    if not ember:
        return f"Ember {ember_id} not found."

    tags_str = ", ".join(ember.tags) if ember.tags else "none"
    source = f"\nSource: {ember.source_path}" if ember.source_path else ""
    return (
        f"🔥 {ember.name} ({ember.importance})\n\n"
        f"{ember.content}\n\n"
        f"Tags: {tags_str}{source}"
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def ember_save_session(
    summary: str,
    decisions: str = "",
    next_steps: str = "",
    source_path: str = "",
) -> str:
    """
    Save key takeaways from the current session. Call before ending a conversation
    where important work was done.

    Args:
        summary: Brief summary of the session's key work
        decisions: Decisions made during the session
        next_steps: Open items and next actions
        source_path: Optional path to the handoff file on disk.
            When provided, enables deep recall to read the full handoff for richer context.
    """
    await _ensure_init()
    saved = []
    resolved_source = source_path if source_path else None

    async def _store_session_ember(name, content, tags, importance):
        vector = await _engine.embed(content)
        cell_id = _engine.assign_cell(vector)
        ember = Ember(
            name=name,
            content=content,
            tags=["session"] + tags,
            cell_id=cell_id,
            importance=importance,
            source="session",
            source_path=resolved_source,
        )
        faiss_id = await _storage.save_ember(ember)
        await _engine.add_vector(faiss_id, vector)
        await _shadow_on_insert(ember, vector.flatten())

    if summary:
        await _store_session_ember("Session Summary", summary, ["summary"], "context")
        saved.append("summary")

    if decisions:
        await _store_session_ember(
            "Session Decisions", decisions, ["decisions"], "decision"
        )
        saved.append("decisions")

    if next_steps:
        await _store_session_ember(
            "Next Steps", next_steps, ["next_steps"], "learning"
        )
        saved.append("next steps")

    return f"Session saved: {', '.join(saved)}. These will be available in your next conversation."


# ---------------------------------------------------------------------------
# MCP Tools: Shadow-Decay Analysis
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_drift_check() -> str:
    """
    Analyze knowledge region health using Shadow-Decay conflict density.
    Reports drifting regions (high shadow accumulation) and silent regions (low topic vitality).
    """
    await _ensure_init()
    config = load_config()

    drifting = []
    silent = []
    healthy = 0

    for cell_id in range(config.k_cells):
        stats = await _storage.get_region_stats(cell_id)
        if not stats:
            silent.append(f"  Cell {cell_id}: no data (uninitialized)")
            continue

        if stats.shadow_accum > 0.3:
            drifting.append(
                f"  Cell {cell_id}: conflict_density={stats.shadow_accum:.3f} (HIGH)"
            )
        elif stats.vitality_score < config.vitality_min:
            silent.append(
                f"  Cell {cell_id}: vitality={stats.vitality_score:.4f} (SILENT)"
            )
        else:
            healthy += 1

    lines = [
        f"Knowledge Region Health ({config.k_cells} cells)",
        "=" * 50,
        f"Healthy: {healthy}  |  Drifting: {len(drifting)}  |  Silent: {len(silent)}",
        "",
    ]

    if drifting:
        lines.append("Drifting regions (high conflict density):")
        lines.extend(drifting)
        lines.append("")

    if silent:
        lines.append("Silent regions (low vitality):")
        lines.extend(silent)
        lines.append("")

    if not drifting and not silent:
        lines.append("All regions healthy. No drift or silence detected.")

    return "\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_graph_search(query: str, depth: int = 2, top_k: int = 5) -> str:
    """
    Vector search → entry node → BFS via knowledge graph edges → return correlated context.

    Args:
        query: What to search for
        depth: How many hops to traverse (default 2)
        top_k: Max results to return (default 5)
    """
    await _ensure_init()
    await _reload_from_disk()

    # Find entry point via HESTIA
    entry_results = await _fetch_and_rerank(query, top_k=1)
    if not entry_results:
        return "No embers found."

    entry_ember, entry_score, _ = entry_results[0]
    entry_id = entry_ember.ember_id

    # BFS traversal from entry point
    connected_ids = await _storage.traverse_kg(entry_id, depth=depth)

    # Load discovered embers
    graph_embers = []
    for eid in connected_ids:
        ember = await _storage.get_ember(eid)
        if ember:
            graph_embers.append(ember)

    # Format output
    lines = [
        f"Graph search for: '{query}'",
        f"Entry: 🔥 {entry_ember.name} [id: {entry_ember.ember_id}] (score: {entry_score:.2f})",
        f"  {_make_preview(entry_ember.content)}",
        "",
        f"Connected memories ({len(graph_embers)} found via {depth}-hop traversal):",
    ]

    for ember in graph_embers[:top_k]:
        shadow_info = f" [shadow:{ember.shadow_load:.1f}]" if ember.shadow_load > 0.1 else ""
        stale_info = " [STALE]" if ember.is_stale else ""
        preview = _make_preview(ember.content)
        lines.append(
            f"  🔥 {ember.name} [id: {ember.ember_id}]{stale_info}{shadow_info}: {preview}"
        )

    if not graph_embers:
        lines.append("  No graph-connected memories found.")

    lines.append("\n→ Use ember_read(id) for full content of any memory.")
    return "\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_health() -> str:
    """
    Compute hallucination risk across all embers, log to metrics, return health report with trend.
    """
    await _ensure_init()
    config = load_config()

    embers = await _storage.list_embers()
    if not embers:
        return "No embers in storage. Health check not applicable."

    # Collect shadow loads and stale flags
    shadow_loads = [e.shadow_load for e in embers]
    stale_flags = [e.is_stale for e in embers]

    # Collect per-cell vitality scores
    vitalities = []
    for cell_id in range(config.k_cells):
        stats = await _storage.get_region_stats(cell_id)
        if stats:
            vitalities.append(stats.vitality_score)
        else:
            vitalities.append(0.0)

    # Compute risk
    risk_data = compute_hallucination_risk(
        shadow_loads, stale_flags, vitalities, config.vitality_min
    )

    # Log metric
    await _storage.log_metric("hallucination_risk", risk_data["risk_score"], risk_data)

    # Get trend
    history = await _storage.get_metric_history("hallucination_risk", limit=5)
    trend_values = [f"{h['value']:.3f}" for h in history]
    trend_str = " → ".join(trend_values) if trend_values else "no history"

    return (
        f"Health: risk={risk_data['risk_score']:.3f} (0=ok, 1=bad) | "
        f"trend: {trend_str}\n"
        f"Total: {risk_data['total']} | "
        f"Shadowed(Φ>0.5): {risk_data['shadowed_count']} | "
        f"Stale: {risk_data['stale_count']} | "
        f"Silent: {risk_data['silent_topics']} | "
        f"Avg Φ: {risk_data['avg_shadow_load']:.3f}"
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def ember_recompute_shadows() -> str:
    """
    Full recalculation of shadow_load for every ember.
    Use after migration or data import.
    """
    await _ensure_init()
    await _reload_from_disk()
    config = load_config()

    embers = await _storage.list_embers()
    if not embers:
        return "No embers to recompute."

    updated = 0
    for ember in embers:
        # Get this ember's FAISS int ID
        int_id = _storage.uuid_to_int.get(ember.ember_id)
        if int_id is None:
            continue

        # Reconstruct vector
        vec = _engine.reconstruct_vector(int_id)
        if vec is None:
            continue

        # Search neighbors
        results = _engine.search(vec.reshape(1, -1), top_k=config.shadow_k + 1)
        if not results:
            continue

        # Build neighbor data
        neighbor_vecs = []
        neighbor_times = []
        neighbor_ids = []

        for faiss_id, dist in results:
            uuid = _storage.int_to_uuid.get(faiss_id)
            if not uuid or uuid == ember.ember_id:
                continue
            neighbor = await _storage.get_ember(uuid)
            if not neighbor:
                continue

            n_vec = _engine.reconstruct_vector(faiss_id)
            if n_vec is not None:
                neighbor_vecs.append(n_vec)
                neighbor_times.append(neighbor.created_at)
                neighbor_ids.append(uuid)

        if not neighbor_vecs:
            continue

        # Compute shadow load
        shadow_load, shadower_id = compute_shadow_load(
            vec,
            ember.created_at,
            neighbor_vecs,
            neighbor_times,
            neighbor_ids,
            config.shadow_delta,
            config.shadow_epsilon,
        )

        # Update if changed
        if abs(shadow_load - ember.shadow_load) > 0.001:
            ember.shadow_load = shadow_load
            ember.shadowed_by = shadower_id
            ember.shadow_updated_at = datetime.now(timezone.utc)
            await _storage.update_ember(ember)

            if shadower_id and shadow_load > 0:
                await _storage.save_edge(
                    ember.ember_id, shadower_id, "shadow", shadow_load
                )

            updated += 1

    return f"Recomputed shadows for {len(embers)} embers. {updated} updated."


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
async def ember_explain(ember_id: str) -> str:
    """
    Return HESTIA score breakdown for a specific ember:
    shadow load, vitality, edges, and scoring factors.

    Args:
        ember_id: The ID of the ember to explain
    """
    await _ensure_init()
    config = load_config()

    ember = await _storage.get_ember(ember_id)
    if not ember:
        return f"Ember {ember_id} not found."

    # Get vector for vitality calculation
    int_id = _storage.uuid_to_int.get(ember_id)
    vitality = 0.0
    if int_id is not None:
        vec = _engine.reconstruct_vector(int_id)
        if vec is not None:
            radius_l2 = VectorEngine.cosine_to_l2(1.0 - config.topic_radius)
            radius_results = _engine.search_radius(vec.reshape(1, -1), radius_l2)
            now = datetime.now(timezone.utc)

            neighbor_times = []
            neighbor_dists = []
            for fid, dist_sq in radius_results:
                uuid = _storage.int_to_uuid.get(fid)
                if uuid:
                    ne = await _storage.get_ember(uuid)
                    if ne:
                        neighbor_times.append(ne.created_at)
                        neighbor_dists.append(dist_sq)

            vitality = compute_topic_vitality(
                neighbor_dists, neighbor_times, now, radius_l2, config.vitality_lambda
            )

    # HESTIA at self-similarity = 1.0 (perfect match)
    v_max = max(vitality, 0.001)
    score, breakdown = compute_hestia_score(
        1.0, ember.shadow_load, vitality, v_max,
        config.shadow_gamma, config.nostalgia_alpha,
    )

    # Get edges
    edges = await _storage.get_edges(ember_id)
    edge_lines = []
    for e in edges:
        other = e["target_id"] if e["source_id"] == ember_id else e["source_id"]
        edge_lines.append(f"  {e['edge_type']}: {other[:8]}... (weight={e['weight']:.2f})")

    lines = [
        f"Ember Explanation: {ember.name}",
        f"ID: {ember_id}",
        "=" * 50,
        "",
        "HESTIA Factors (at perfect query match):",
        f"  Final Score: {score:.4f}",
        f"  Cosine Sim: 1.0 (self)",
        f"  Shadow Factor: {breakdown['shadow_factor']:.4f}  (shadow_load={ember.shadow_load:.3f})",
        f"  Vitality Factor: {breakdown['vitality_factor']:.4f}  (vitality={vitality:.3f})",
        "",
        f"Shadow Load: {ember.shadow_load:.3f}",
        f"Shadowed By: {ember.shadowed_by or 'None'}",
        f"Related IDs: {', '.join(ember.related_ids) if ember.related_ids else 'None'}",
        f"Stale: {ember.is_stale} ({ember.stale_reason or 'N/A'})",
        "",
        f"Edges ({len(edges)}):",
    ]
    lines.extend(edge_lines if edge_lines else ["  None"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
def start_session() -> str:
    """Load memory and context at the start of a new conversation."""
    return """Check my persistent memory for any relevant context.

Steps:
1. Call ember_auto with a summary of what the user is asking about
2. If relevant memories are found, incorporate them naturally
3. If there are recent "Next Steps" embers, mention what was planned
4. Respond to the user with full context, as if you remember everything"""


@mcp.prompt()
def end_session() -> str:
    """Save important context before ending a conversation."""
    return """Before we end this conversation, let's save the important parts.

Steps:
1. Summarize the key work done in this session (2-3 sentences)
2. List any decisions that were made
3. Note any next steps or open items
4. Call ember_save_session with all three
5. Confirm to the user what was saved"""


@mcp.prompt()
def remember() -> str:
    """Store something the user wants remembered across sessions."""
    return """The user wants to save something to persistent memory.

Steps:
1. Identify what needs to be remembered (preference, fact, rule, context)
2. Choose a clear, searchable name
3. Add relevant tags for categorization
4. Choose the right importance level: fact, decision, preference, context, or learning
5. Call ember_store with the structured content and importance
6. Confirm what was saved"""
