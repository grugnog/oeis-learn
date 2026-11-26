"""Training-only evidence for the diagnostic funding predicate."""
from oeis_learn.generalization.progressive.training_rows import verified_training_pairs


def training_support(donor, references):
    missing = dict(status="unavailable", eligible_rows=0, eligible_families=0, positions=0, pairs=[])
    ref = references.get("private/channels/screening-training")
    if ref is None:
        return missing
    if (ref["kind"], ref["role"]) != ("prepared", "learner"):
        raise ValueError("training support is outside the registered learner role")
    channel = donor.optional(donor.bound, ref["artifact"], version="discovery-channel/v1")
    if channel is None:
        return missing
    if (channel["role"], channel["kind"], channel["data"]["membership"]) != ("learner", "training_request", "training"):
        raise ValueError("isolated training channel required")
    package = donor.optional(donor.bound, channel["data"]["artifact"], version="progressive-training-input/v1")
    if package is None:
        return missing
    if package["scope"] != "NATIVE_VERIFIED":
        raise ValueError("mechanical native training provenance required")
    return dict(status="measured", input=package["identity"], **verified_training_pairs(package["shards"]))
