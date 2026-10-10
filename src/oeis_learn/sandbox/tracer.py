"""Real reference events; unsupported native/token-causal traces fail explicitly."""

from oeis_learn.data.models import FineGrainedAttributionSpan


def classify_failure_mode(exec_result, target_terms):
    if exec_result.status in ("PARSE_ERROR", "COMPILE_ERROR", "MISSING_ENTRYPOINT", "CONFIG_ERROR"):
        return "SYNTAX"
    if exec_result.status in ("OUT_OF_FUEL", "EXECUTION_TRAP"):
        return "CONSTRAINT"
    if exec_result.status == "SUCCESS" and list(exec_result.output) == list(target_terms):
        return "CORRECT"
    return "LOGIC"


def locate_divergence_token_span(*args, **kwargs):
    raise ValueError("unsupported: term divergence does not establish a causal token span")


def locate_causal_divergence_span(*args, **kwargs):
    raise ValueError("unsupported: no instrumented native causal trace")


def extract_token_coverage(*args, **kwargs):
    raise ValueError("unsupported: text is not runtime coverage")


def build_fine_grained_attribution(
    wat_code, exec_result, target_terms, total_advantage, total_tokens
):
    # Whole-trajectory weights are an objective policy, not execution attribution.
    return FineGrainedAttributionSpan(
        classify_failure_mode(exec_result, target_terms),
        exec_result.divergence_step,
        0,
        0,
        [total_advantage / max(1, total_tokens)] * total_tokens,
        [],
    )


def trace_program(source, index, *, mode="reference", max_events=10000):
    import time
    from oeis_learn.sandbox.wat_ast import parse
    from oeis_learn.sandbox.lowering import lower_with_map
    from oeis_learn.sandbox.reference import Interpreter
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    if mode != "reference":
        return {"status": "unsupported", "reason": "native runtime events are not instrumented"}
    if type(max_events) is not int or not 1 <= max_events <= 10000:
        raise ValueError("bounded trace event cap required")
    if type(index) is not int or not 0 <= index <= 99:
        raise ValueError("exact rebased index required")
    program = parse(source)
    lowered, mapping, nodes = lower_with_map(program)
    events = []

    def event(node, step, stack):
        if len(events) >= max_events:
            from oeis_learn.sandbox.wat_ast import ExecutionFailure

            raise ExecutionFailure("execution_limit", "trace_event_cap")
        node_id = nodes[id(node)]
        events.append({"instruction_id": node_id, "step": step, "stack_before": list(stack)})

    interpreter = Interpreter(1000000, time.monotonic_ns() + 2000000000, event=event)
    result = interpreter.run(program, index)
    return {
        "version": "reference-trace/v1",
        "status": "complete" if result.outcome is None else "failed",
        "source_sha256": digest(source),
        "lowered_sha256": digest(lowered),
        "mapping": mapping,
        "events": events,
        "value": None if result.value is None else str(result.value),
        "outcome": result.outcome,
        "reason": result.reason,
        "coverage": sorted({e["instruction_id"] for e in events}),
    }
