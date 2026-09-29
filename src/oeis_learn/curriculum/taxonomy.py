"""5-Stage Mathematical Taxonomy Mapping and Difficulty Weighting."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set

TAXONOMY_STAGE_NAMES: Dict[int, str] = {
    1: "Primitives & Polynomials",
    2: "Linear Recurrences & Rational GFs",
    3: "Holonomic & D-Finite Sequences",
    4: "Combinatorics & Number Theory",
    5: "Exhaustive Search & Graph Invariants",
}

STAGE_TAG_MAP: Dict[int, Set[str]] = {
    1: {"core", "easy", "nonn", "polynomial"},
    2: {"core", "frac", "cons", "mult", "fibonacci", "lucas"},
    3: {"nice", "cofr", "tabl", "tabf", "factorial", "catalan"},
    4: {"hard", "base", "eigen", "prime", "partition"},
    5: {"hard", "bref", "more", "graph", "search"},
}


def get_stage_name(stage: int) -> str:
    return TAXONOMY_STAGE_NAMES.get(stage, f"Stage {stage}")


def compute_prompt_difficulty_weight(tags: Sequence[str], stage: int, term_count: int) -> float:
    """Computes a normalized difficulty weighting w_x for a given prompt in competence calculation."""
    base_weight = 1.0 + (stage - 1) * 0.2
    tag_set = {t.lower() for t in tags}

    if "hard" in tag_set:
        base_weight += 0.3
    if "core" in tag_set:
        base_weight += 0.1
    if "bref" in tag_set:
        base_weight += 0.4

    return float(base_weight)
