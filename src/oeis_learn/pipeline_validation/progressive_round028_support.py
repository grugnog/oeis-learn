"""Training-only evidence for the diagnostic funding predicate."""
from oeis_learn.generalization.progressive.training_rows import held_out_probes, verified_training_pairs


def training_package(donor, references):
    ref = references.get("private/channels/screening-training")
    if ref is None:
        return None
    if (ref["kind"], ref["role"]) != ("prepared", "learner"):
        raise ValueError("training support is outside the registered learner role")
    channel = donor.optional(donor.bound, ref["artifact"], version="discovery-channel/v1")
    if channel is None:
        return None
    if (channel["role"], channel["kind"], channel["data"]["membership"]) != ("learner", "training_request", "training"):
        raise ValueError("isolated training channel required")
    package = donor.optional(donor.bound, channel["data"]["artifact"], version="progressive-training-input/v1")
    if package is None:
        return None
    if package["scope"] != "NATIVE_VERIFIED":
        raise ValueError("mechanical native training provenance required")
    return package


def training_support(donor, references, *, legacy=False):
    package = training_package(donor, references)
    if package is None:
        return dict(status="unavailable", eligible_rows=0, eligible_families=0, positions=0, pairs=[])
    if legacy:
        return dict(status="measured", input=package["identity"], **verified_training_pairs(package["shards"]))
    pairs = verified_training_pairs(package["shards"], excluded=held_out_probes(package["layout"]))
    return dict(status="measured", input=package["identity"], eligibility="executable-training/v1", **pairs)
