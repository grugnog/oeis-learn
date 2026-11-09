"""Frozen complete-iteration bounds protect the finalization reserve."""


def margin(payload, *, research=False):
    value = payload.get("iteration_budget")
    if value is None:
        if research:
            raise ValueError("measured complete iteration budget required")
        return 0
    if set(value) != {"update_us", "probe_us", "checkpoint_us", "producer"}:
        raise ValueError("closed iteration budget required")
    phases = [value[k] for k in ("update_us", "probe_us", "checkpoint_us")]
    if any(type(n) is not int or n <= 0 for n in phases):
        raise ValueError("positive frozen iteration bounds required")
    return sum(phases)
