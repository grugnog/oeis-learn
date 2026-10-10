"""Publish explicit G8 extensions without mutating the frozen foundation control."""

from copy import deepcopy
from pathlib import Path
from oeis_learn.experiments.artifacts import (
    compute_canonical_digest as digest,
    canonical_bytes,
    atomic_write,
    ArtifactPath,
)
from oeis_learn.experiments.qualified_models import seal_record, validate_record
from oeis_learn.experiments.profiles import qualified_model_profile
from oeis_learn.data.structural_splits import StructuralHoldouts


def holdout_profile(holdouts):
    return {
        "version": "expanded-holdouts/v1",
        "seed": holdouts.seed,
        "structures": sorted(holdouts.reserved_structures),
        "compositions": [sorted(v) for v in holdouts.reserved_compositions],
        "parameter_bits": holdouts.reserved_parameter_bits,
        "prefixes": sorted(holdouts.reserved_prefixes),
    }


def load_holdouts(value):
    if (
        set(value)
        != {"version", "seed", "structures", "compositions", "parameter_bits", "prefixes"}
        or value["version"] != "expanded-holdouts/v1"
    ):
        raise ValueError("closed expanded holdout profile required")
    holdouts = StructuralHoldouts(
        seed=value["seed"],
        reserved_structures=frozenset(value["structures"]),
        reserved_compositions=tuple(frozenset(v) for v in value["compositions"]),
        reserved_parameter_bits=value["parameter_bits"],
        reserved_prefixes=frozenset(value["prefixes"]),
    )
    if holdout_profile(holdouts) != value:
        raise ValueError("canonical holdout sets required")
    return holdouts


def extend_learning_contract(base, report, output, *, holdouts, model=None):
    validate_record(base)
    if (
        base["kind"] != "qualified_contract"
        or base["gates"]["G7"] is None
        or report["gate"] != "G8"
        or report["software_status"] != "PASS"
    ):
        raise ValueError("G7 base and G8 software evidence required; activation replays both")
    model = qualified_model_profile() if model is None else model
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)

    def publish(value):
        key = digest(value, "identity_sha256" if "identity_sha256" in value else None)
        atomic_write(ArtifactPath(root, key[7:] + ".json"), canonical_bytes(value))
        return key

    profile_ids = deepcopy(base["profiles"])
    profile_ids["model"] = publish(model)
    profile_ids["objective"] = publish(
        {"version": "objective/v1", "objective": model["objective"], "policy": model["policy"]}
    )
    profile_ids["sampling"] = publish({"version": "sampling/v1", "policy": model["policy"]})
    profile_ids["admission"] = publish(holdout_profile(holdouts))
    modes = deepcopy(base["modes"])
    settings = {
        "learning": {"model_profile": model},
        "replay": {"fraction_percent": 25, "implementations_per_output": 4},
        "search": {
            "portfolio": "portfolio32",
            "attempts": 32,
            "stochastic": 16,
            "beam": 8,
            "repair": 8,
        },
    }
    for name, value in settings.items():
        key = publish({"version": "qualified-service/v1", "service": name, "settings": value})
        modes[name] = {"enabled": True, "profile_sha256": key}
        if name == "search":
            profile_ids["search"] = key
    gates = dict(base["gates"], G8=report["report_sha256"])
    return seal_record(
        "qualified_contract",
        base_contract_sha256=base["base_contract_sha256"],
        profiles=profile_ids,
        gates=gates,
        track="strict_generic",
        initialization=base["initialization"],
        parent_chain=base["parent_chain"],
        objective=model["objective"],
        modes=modes,
    )
