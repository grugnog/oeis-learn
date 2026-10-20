"""Read-only historical evidence; no retired solver imports or status promotion."""

from copy import deepcopy


def inspect_record(record):
    if not isinstance(record, dict):
        raise ValueError("historical record must be an object")
    return dict(
        status="historical_unreplayed",
        original=deepcopy(record),
        reason="Retired fitting evidence is retained for inspection, not executable replay or training admission.",
    )
