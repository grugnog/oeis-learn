"""Analysis-only external lookup after immutable proposal sealing."""

from copy import deepcopy
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


def seal_proposal(proposal):
    record = deepcopy(proposal)
    if "proposal_sha256" in record:
        raise ValueError("proposal already sealed")
    record["proposal_sha256"] = digest(record)
    return record


def validate_seal(record):
    if record.get("proposal_sha256") != digest(record, "proposal_sha256"):
        raise ValueError("frozen proposal changed")


def analyze_novelty(proposal, lookup, *, role):
    if role != "analysis":
        raise ValueError("external knowledge restricted to post-seal analysis role")
    validate_seal(proposal)
    frozen = deepcopy(proposal)
    identity = proposal["proposal_sha256"]
    # Never give a lookup callback a mutable reference to the stored proposal.
    knowledge = lookup(deepcopy(frozen))
    validate_seal(proposal)
    if proposal != frozen:
        raise ValueError("lookup altered frozen proposal")
    if knowledge is None:
        classification = "novelty_unknown"
    elif (
        not isinstance(knowledge, dict)
        or set(knowledge) != {"relation_known", "implementation_known", "source_reference"}
        or any(type(knowledge[k]) is not bool for k in ("relation_known", "implementation_known"))
        or not isinstance(knowledge["source_reference"], str)
        or not knowledge["source_reference"]
    ):
        raise ValueError("typed post-hoc knowledge evidence required")
    elif knowledge["implementation_known"]:
        classification = "rediscovery"
    elif knowledge["relation_known"]:
        classification = "new_implementation"
    else:
        classification = "novelty_unknown"
    return {
        "version": "novelty-analysis/v1",
        "proposal_sha256": identity,
        "classification": classification,
        "relation_status": proposal.get("evidence_kind", "empirical_relation"),
        "knowledge": deepcopy(knowledge),
        "role": "analysis",
        "training_eligible": False,
        "identity_sha256": digest([identity, classification, knowledge]),
    }
