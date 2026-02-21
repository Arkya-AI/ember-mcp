import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


def utc_now() -> datetime:
    """Factory function for current UTC datetime."""
    return datetime.now(timezone.utc)


class Ember(BaseModel):
    """
    Represents a unit of memory with temporal intelligence capabilities.
    """
    ember_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    content: str
    tags: List[str] = Field(default_factory=list)
    cell_id: int = -1

    # Temporal intelligence
    importance: str = "context"  # fact, decision, preference, context, learning
    supersedes_id: Optional[str] = None
    is_stale: bool = False
    stale_reason: Optional[str] = None
    status: Optional[str] = None  # None | open | in_progress | done

    # Access tracking
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0

    # Session and source tracking
    session_id: Optional[str] = None
    source: str = "manual"  # manual, auto, session
    source_path: Optional[str] = None

    # Shadow-Decay fields
    shadow_load: float = Field(default=0.0, ge=0.0, le=1.0)  # Φᵢ(t): cached shadow load
    shadowed_by: Optional[str] = None  # ember_id of dominant shadower
    shadow_updated_at: Optional[datetime] = None
    related_ids: List[str] = Field(default_factory=list)  # KG edges (max 5)
    superseded_by_id: Optional[str] = None  # Reverse link from contradict

    # Compaction tracking
    is_compacted: bool = False
    original_content_length: Optional[int] = None  # chars before compaction

    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)


class EmberMetadata(BaseModel):
    """Lightweight representation of an Ember for list views."""
    ember_id: str
    name: str
    tags: List[str]
    cell_id: int
    importance: str
    is_stale: bool
    status: Optional[str] = None
    is_compacted: bool = False
    created_at: datetime

    @classmethod
    def from_ember(cls, ember: "Ember") -> "EmberMetadata":
        return cls(
            ember_id=ember.ember_id,
            name=ember.name,
            tags=ember.tags,
            cell_id=ember.cell_id,
            importance=ember.importance,
            is_stale=ember.is_stale,
            status=ember.status,
            is_compacted=ember.is_compacted,
            created_at=ember.created_at,
        )


class SearchResult(BaseModel):
    """Search result with ember and temporal score."""
    ember: Ember
    score: float


class EmberConfig(BaseModel):
    """Global configuration for the Ember server."""
    k_cells: int = 16
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    similarity_threshold: float = 0.4
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".ember")

    # Shadow-Decay configuration
    shadow_delta: float = 0.3           # Shadow cone aperture
    shadow_epsilon: float = 0.05        # Min similarity offset (anti-self-shadow)
    shadow_gamma: float = 2.0           # Shadow hardness exponent
    nostalgia_alpha: float = 0.1        # Floor for silent topic retrieval
    topic_radius: float = 0.5           # R_topic for vitality (cosine distance)
    vitality_lambda: float = 0.05       # Decay rate per day
    vitality_min: float = 0.01          # V_min silence threshold
    shadow_k: int = 10                  # k neighbors checked on insert

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HallucinationMetrics(BaseModel):
    """Hallucination risk metrics across the memory store."""
    total: int
    shadowed_count: int          # embers with shadow_load > 0.5
    stale_count: int             # embers with is_stale=True
    silent_topics: int           # topic regions with vitality < V_min
    avg_shadow_load: float
    risk_score: float            # 0-1 composite
    timestamp: datetime = Field(default_factory=utc_now)


class ShadowEdge(BaseModel):
    """Represents a shadowing relationship between two embers."""
    source_id: str               # the shadowed ember
    shadower_id: str             # the newer ember doing the shadowing
    phi: float                   # shadow potential value
    cosine_sim: float
    detected_at: datetime = Field(default_factory=utc_now)


class RegionStats(BaseModel):
    """Lightweight per-cell volatility tracking via Shadow-Decay signals."""
    cell_id: int
    vitality_score: float = 0.0       # V(centroid, t) — topic activity level
    shadow_accum: float = 0.0         # EMA of shadow potentials — conflict density
    last_updated: datetime = Field(default_factory=utc_now)


# Decay half-lives by importance (in days)
DECAY_HALF_LIVES = {
    "fact": 365.0,
    "decision": 30.0,
    "preference": 60.0,
    "context": 7.0,
    "learning": 90.0,
}

# Valid edge types for the knowledge graph
VALID_EDGE_TYPES = {
    "shadow", "related", "supersedes",        # system-managed
    "depends_on", "child_of", "context_for",  # user-specifiable
}
USER_EDGE_TYPES = {"depends_on", "child_of", "context_for"}
