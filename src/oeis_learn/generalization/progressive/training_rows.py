"""Shared closed learner-row validation for preparation and installed training."""
from .contracts import seal


def validate_row(row, *, native_verified, short=False):
    from oeis_learn.decoder.bigint_codec import encode_body

    fields = {
        "version",
        "identity",
        "ordinal",
        "draw_identity",
        "source_policy",
        "source",
        "structure",
        "behavior",
        "indices",
        "values",
        "target_source",
        "tokens",
        "target_disposition",
        "membership",
        "native_verified",
        "control_events",
        "branch_events",
        "operations",
        "detector_limit",
    }
    if set(row) != fields:
        raise ValueError("closed training row; evaluation metadata excluded")
    if (
        seal(row) != row
        or row.get("version") != "progressive-training-row/v1"
        or row.get("membership") != "training"
    ):
        raise ValueError("sealed training rows only")
    if native_verified and row.get("native_verified") is not True:
        raise ValueError("native verification missing")
    if (
        len(row["values"]) != (4 if short else 100)
        or len(row["indices"]) != (4 if short else 100)
        or encode_body(row["target_source"]) != row["tokens"]
    ):
        raise ValueError("complete target/codec binding required")
    return row
