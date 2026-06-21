"""
engine/fusion.py

LAYER 3 — Fusion + Stuffer Detection.

raw_score    = semantic*w_sem + skill*w_skill + career*w_career
final_score  = raw_score * behavioral_multiplier * integrity_multiplier * stuffer_penalty

Behavioral and integrity are explicitly NOT part of the additive raw_score
— they are multiplicative gates on top of it, modeling the real-world
truth that an unavailable or fabricated candidate is disqualifying
regardless of how good their raw qualifications look.

Stuffer detection compares skill_embedding-vs-JD-query similarity against
career_embedding-vs-JD-query similarity for the same candidate. A large
gap (skills inflated relative to career evidence) triggers a penalty.

RAMPED PENALTY (Item 6b): the penalty is no longer a binary cliff at
gap > threshold. It's a smooth linear ramp:
  gap ≤ gap_threshold                   → penalty = 1.0 (no penalty)
  gap ∈ (threshold, threshold+ramp_width] → linear ramp: 1.0 → penalty_multiplier
  gap > threshold + ramp_width           → flat penalty_multiplier

This eliminates the 3.3× score cliff that existed between candidates at
gap=0.39 vs gap=0.41. Default ramp_width=0.15 (config: stuffer_ramp_width).

This generalizes to any JD because it never references a known stuffer
list — it just compares two views of the same person.
"""

from __future__ import annotations
import numpy as np


def compute_stuffer_penalty(
    skill_sim_mean: np.ndarray,
    career_sim_mean: np.ndarray,
    gap_threshold: float,
    penalty_multiplier: float,
    ramp_width: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    skill_sim_mean, career_sim_mean: (N,) arrays — mean cosine sim of
    skill_embedding / career_embedding against JD query embeddings,
    rescaled to [0,1] the same way channel_semantic does.

    gap_threshold:     gap below which no penalty applies
    penalty_multiplier: penalty at gap ≥ threshold + ramp_width (floor)
    ramp_width:        width of the linear ramp zone (default 0.15)

    Returns: ((N,) stuffer_penalty array (1.0=no penalty), (N,) gap array)
    """
    gap = skill_sim_mean - career_sim_mean
    # t = 0 at threshold, t = 1 at threshold + ramp_width
    t = np.clip((gap - gap_threshold) / max(ramp_width, 1e-6), 0.0, 1.0)
    # Linear interpolation: 1.0 (no penalty) → penalty_multiplier
    penalty = 1.0 - t * (1.0 - penalty_multiplier)
    return penalty, gap


def fuse(
    semantic_scores: np.ndarray,
    skill_scores: np.ndarray,
    career_scores: np.ndarray,
    behavioral_multipliers: np.ndarray,
    integrity_multipliers: np.ndarray,
    stuffer_penalties: np.ndarray,
    weights: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    All inputs are (N,) numpy arrays aligned by candidate index.
    weights: dict with keys semantic, skills, career (must sum ~1.0 for the
    additive layer; behavioral/integrity weights in config are informational
    only since they're applied multiplicatively here, not as linear terms).
    """
    w_sem = weights.get("semantic", 0.30)
    w_skill = weights.get("skills", 0.20)
    w_career = weights.get("career", 0.20)

    raw_score = (semantic_scores * w_sem) + (skill_scores * w_skill) + (career_scores * w_career)
    final_score = raw_score * behavioral_multipliers * integrity_multipliers * stuffer_penalties
    return raw_score, final_score
