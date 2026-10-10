"""Finite compiled/reference observations, separate from class-specific proofs.

Called only inside the bounded certificate worker. This checks a small declared
machine-domain sample and full successful state, never a formal compiler theorem.
"""

from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.sandbox.streaming import LogicalRuntime, LogicalLimits, identity
from oeis_learn.sandbox.wat_ast import ExecutionFailure


def finite_compiled_conformance(programs, domain):
    start = max(-(1 << 31), domain["start"])
    end = min((1 << 31) - 1, domain["end"] if domain["end"] is not None else (1 << 31) - 1)
    indices = []
    for index in range(start, min(end + 1, start + 1030)):
        if index not in domain["excluded_indices"]:
            indices.append(index)
            if len(indices) == 5:
                break
    if not indices:
        return {
            "status": "unavailable_outside_machine_domain",
            "programs": [],
            "wasm_theorem": False,
        }
    records = []
    for source in programs:
        runtime = LogicalRuntime(source, LogicalLimits())
        observations = []

        def observe(index, action):
            try:
                value = action()
                if runtime.wasm_state() != runtime.ref.state():
                    raise ValueError("compiled/reference state disagreement")
                observations.append(
                    {
                        "index": index,
                        "outcome": None,
                        "value": None if value is None else str(value),
                        "state_sha256": digest(runtime.wasm_state()),
                    }
                )
                return True
            except ExecutionFailure as exc:
                if exc.outcome == "execution_limit":
                    raise TimeoutError("finite compiler conformance limit") from exc
                if "disagreement" in exc.reason:
                    raise ValueError(exc.reason) from exc
                # pair()/initialize()/next() already require both independent
                # engines to agree on trap classification; failed state is not
                # promoted to a successful transition.
                observations.append(
                    {"index": index, "outcome": exc.outcome, "value": None, "state_sha256": None}
                )
                return False

        if runtime.program.profile == "pure":
            for index in indices:
                runtime.fresh()
                observe(index, lambda: runtime.pair("compute", "compute", index))
        else:
            if start != 0 or domain["excluded_indices"]:
                raise NotImplementedError("finite stream compiler conformance requires origin zero")
            if observe(0, lambda: runtime.initialize(0)):
                for index in indices:
                    if not observe(index, runtime.next):
                        break
        records.append(
            {
                "identity": identity(source),
                "observations": observations,
                "fuel": runtime.fuel,
                "reference_steps": runtime.steps,
            }
        )
    return {"status": "finite_conformance", "programs": records, "wasm_theorem": False}
