import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

from ember.models import DECAY_HALF_LIVES, RegionStats


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_temporal_score(ember, query_distance: float, now: datetime = None) -> float:
    """Legacy temporal scoring — kept as fallback."""
    if now is None:
        now = datetime.now(timezone.utc)
    semantic_score = max(0.0, 1.0 - (query_distance / 2.0))
    age_days = max(0.0, (now - ember.updated_at).total_seconds() / 86400.0)
    half_life = DECAY_HALF_LIVES.get(ember.importance, 30.0)
    recency_weight = 0.5 ** (age_days / half_life)
    access_boost = math.log(ember.access_count + 1) * 0.1
    staleness_factor = 0.1 if ember.is_stale else 1.0
    score = semantic_score * recency_weight * (1.0 + access_boost) * staleness_factor
    return round(score, 4)


# ---------------------------------------------------------------------------
# Shadow-Decay Framework
# ---------------------------------------------------------------------------


def compute_shadow_potential(
    cos_sim: float,
    t_target: datetime,
    t_shadower: datetime,
    delta: float = 0.3,
    epsilon: float = 0.05,
) -> float:
    """
    Computes the shadowing potential φ(m_shadower | m_target).

    φ(mᵢ | mⱼ) = 𝟙(tⱼ > tᵢ) · max(0, (cos_sim - (1-δ)) / (δ - ε)), clamped [0,1]

    Only newer memories can shadow older ones.
    """
    if t_shadower <= t_target:
        return 0.0

    val = (cos_sim - (1.0 - delta)) / (delta - epsilon)
    return max(0.0, min(1.0, val))


def compute_shadow_load(
    ember_vec: np.ndarray,
    ember_time: datetime,
    neighbor_vecs: List[np.ndarray],
    neighbor_times: List[datetime],
    neighbor_ids: List[str],
    delta: float = 0.3,
    epsilon: float = 0.05,
) -> Tuple[float, Optional[str]]:
    """
    Computes the maximum shadow load Φ on a specific ember from its neighbors.

    Φᵢ = sup{φ(mᵢ | mⱼ) : j ≠ i}

    Returns:
        (max_shadow_load, ember_id_of_dominant_shadower) or (0.0, None) if no shadowing.
    """
    max_load = 0.0
    dominant_shadower_id = None

    for n_vec, n_time, n_id in zip(neighbor_vecs, neighbor_times, neighbor_ids):
        sim = cosine_similarity(ember_vec, n_vec)
        # ember is target, neighbor is potential shadower
        potential = compute_shadow_potential(sim, ember_time, n_time, delta, epsilon)

        if potential > max_load:
            max_load = potential
            dominant_shadower_id = n_id

    return max_load, dominant_shadower_id


def compute_topic_vitality(
    distances_l2_sq: List[float],
    neighbor_times: List[datetime],
    now: datetime,
    radius_l2: float,
    lambda_decay: float = 0.05,
) -> float:
    """
    Computes Topic Vitality V(x, t).

    V(x,t) = Σ 𝟙(d²_k < R²) · exp(-λ · age_days_k)

    Args:
        distances_l2_sq: Squared L2 distances from FAISS.
        neighbor_times: Created_at times of neighbors.
        now: Current UTC datetime.
        radius_l2: L2 distance radius threshold (NOT squared — squared internally).
        lambda_decay: Decay rate per day.

    Returns:
        Vitality score (float >= 0).
    """
    radius_threshold = radius_l2 * radius_l2
    vitality_sum = 0.0

    for dist_sq, t_neighbor in zip(distances_l2_sq, neighbor_times):
        if dist_sq < radius_threshold:
            age_days = max(0.0, (now - t_neighbor).total_seconds() / 86400.0)
            vitality_sum += math.exp(-lambda_decay * age_days)

    return vitality_sum


def compute_hestia_score(
    cos_sim: float,
    shadow_load: float,
    vitality: float,
    v_max: float,
    gamma: float = 2.0,
    alpha: float = 0.1,
) -> Tuple[float, dict]:
    """
    Computes the HESTIA retrieval score.

    S = cos_sim · (1 - Φ)^γ · (α + (1-α) · V/V_max)

    Returns:
        (score, breakdown_dict) with keys: cos_sim, shadow_factor, vitality_factor.
    """
    shadow_factor = (1.0 - shadow_load) ** gamma

    if v_max > 0:
        vitality_factor = alpha + (1.0 - alpha) * (vitality / v_max)
    else:
        vitality_factor = alpha

    score = cos_sim * shadow_factor * vitality_factor

    return score, {
        "cos_sim": cos_sim,
        "shadow_factor": shadow_factor,
        "vitality_factor": vitality_factor,
    }


def compute_hallucination_risk(
    shadow_loads: List[float],
    stale_flags: List[bool],
    vitalities: List[float],
    v_min: float = 0.01,
) -> dict:
    """
    Estimates hallucination risk based on shadow loads, staleness, and topic silence.

    risk = 0.4 × heavily_shadowed + 0.3 × stale_ratio + 0.3 × silent_ratio
    """
    total = len(shadow_loads)
    if total == 0:
        return {
            "total": 0,
            "shadowed_count": 0,
            "stale_count": 0,
            "silent_topics": 0,
            "avg_shadow_load": 0.0,
            "risk_score": 0.0,
        }

    shadowed_count = sum(1 for load in shadow_loads if load > 0.5)
    stale_count = sum(1 for is_stale in stale_flags if is_stale)
    silent_topics = sum(1 for v in vitalities if v < v_min)

    avg_shadow = sum(shadow_loads) / total

    heavily_shadowed_ratio = shadowed_count / total
    stale_ratio = stale_count / total
    silent_ratio = silent_topics / len(vitalities) if vitalities else 0.0

    risk = (0.4 * heavily_shadowed_ratio) + (0.3 * stale_ratio) + (0.3 * silent_ratio)

    return {
        "total": total,
        "shadowed_count": shadowed_count,
        "stale_count": stale_count,
        "silent_topics": silent_topics,
        "avg_shadow_load": round(avg_shadow, 4),
        "risk_score": round(risk, 4),
    }


def detect_kg_edges(
    cos_sims: List[float],
    shadow_potentials: List[float],
    neighbor_ids: List[str],
    threshold: float = 0.4,
    max_edges: int = 5,
) -> List[str]:
    """
    Identifies Knowledge Graph edges: similar but NOT shadowing neighbors.

    Criteria: cos_sim > threshold AND φ < 0.1
    Returns neighbor_ids sorted by similarity descending, capped at max_edges.
    """
    candidates = []
    for sim, phi, nid in zip(cos_sims, shadow_potentials, neighbor_ids):
        if sim > threshold and phi < 0.1:
            candidates.append((sim, nid))

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    return [c[1] for c in candidates[:max_edges]]


def update_region_stats(
    cell_id: int,
    phi_value: float,
    existing_stats: Optional[RegionStats] = None,
    ema_alpha: float = 0.1,
) -> RegionStats:
    """
    Updates per-cell conflict density using Exponential Moving Average.

    shadow_accum_new = (1 - α) · old_accum + α · φ

    High shadow_accum = high conflict density = region is drifting.
    """
    if existing_stats is None:
        return RegionStats(cell_id=cell_id, shadow_accum=phi_value, vitality_score=0.0)

    new_accum = (1.0 - ema_alpha) * existing_stats.shadow_accum + ema_alpha * phi_value

    return RegionStats(
        cell_id=cell_id,
        shadow_accum=new_accum,
        vitality_score=existing_stats.vitality_score,
    )
