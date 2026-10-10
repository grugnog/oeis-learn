"""Bounded reachable-state closure/cycle certificates, never output heuristics."""

from oeis_learn.sandbox.logical_ir import parse_logical, Node
from oeis_learn.sandbox.reference import LogicalReference
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


def check_state(programs, coefficients, domain, *, max_states=10000):
    if not 1 <= max_states <= 10000:
        raise ValueError("state cap10000")
    if (
        domain["start"] != 0
        or domain.get("excluded_indices")
        or type(domain["end"]) is not int
        or not 0 <= domain["end"] <= (1 << 31) - 2
    ):
        raise NotImplementedError("state origin zero with bounded signed-i32 driver domain")

    def has_index(n):
        return isinstance(n, Node) and (n.op == "index" or any(has_index(a) for a in n.args))

    refs = []
    for source in programs:
        p = parse_logical(source)
        if p.profile != "stream" or has_index(p.root):
            raise NotImplementedError("index-independent finite-state stream required")
        r = LogicalReference(p)
        r.init(0)
        refs.append(r)
    seen = {}
    states = []
    outputs = []
    for step in range(max_states + 1):
        key = tuple((tuple(r.slots), tuple(r.array)) for r in refs)
        if key in seen:
            return all(sum(c * v for c, v in zip(coefficients, row)) == 0 for row in outputs), {
                "states_sha256": digest(states),
                "outputs": [[str(v) for v in row] for row in outputs],
                "cycle_start": seen[key],
                "cycle_length": step - seen[key],
                "reachable_states": len(states),
                "closure": "exact_data_state_with_bounded_driver_index",
                "driver_index_domain": [0, (1 << 31) - 2],
            }
        if step == max_states:
            raise TimeoutError("reachable-state cap; closure unknown")
        seen[key] = step
        states.append(digest(key))
        outputs.append([r.next() for r in refs])
    raise AssertionError("unreachable")
