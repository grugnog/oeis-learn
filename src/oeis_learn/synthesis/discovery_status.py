"""Separate evidence labels; no theorem/novelty inference from finite agreement."""

from .state import require_hash


def validate_discovery(record):
    fields = {
        "finite_match",
        "conjecture",
        "bounded_verification",
        "proved_statement",
        "novelty_review",
        "historical_status",
    }
    if set(record) != fields:
        raise ValueError("separate discovery fields required")
    for key in ("finite_match", "conjecture", "bounded_verification"):
        if record[key] is not None:
            require_hash(record[key])
    proof = record["proved_statement"]
    if proof is not None:
        if (
            not isinstance(proof, dict)
            or set(proof)
            != {
                "statement",
                "certificate",
                "checker",
                "assumptions",
                "domain",
                "independently_checked",
            }
            or proof["independently_checked"] is not True
            or not proof["assumptions"]
            or not proof["domain"]
        ):
            raise ValueError("scoped independently checked certificate required")
        for key in ("statement", "certificate", "checker"):
            require_hash(proof[key])
    novelty = record["novelty_review"]
    if novelty is not None:
        if (
            not isinstance(novelty, dict)
            or set(novelty) != {"review", "comparison", "status"}
            or novelty["status"] not in {"pending", "known", "candidate_novelty"}
        ):
            raise ValueError("separate knowledge-comparison review required")
        require_hash(novelty["review"])
        require_hash(novelty["comparison"])
    if record["historical_status"] not in {None, "historical_unreplayed"}:
        raise ValueError("unknown historical evidence status")
    import copy

    return copy.deepcopy(record)
