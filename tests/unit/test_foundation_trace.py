import pytest
from oeis_learn.sandbox.tracer import trace_program, extract_token_coverage


def test_trace_records_executed_branch_and_maps_lowered_sites():
    source = (
        "i32.const 1 ( if ( result i64 i64 i64 i64 ) ( then i256.const 3 ) ( else i256.const 7 ) )"
    )
    trace = trace_program(source, 0)
    assert trace["status"] == "complete" and trace["value"] == "3"
    mapped = {row["instruction_id"]: row for row in trace["mapping"]}
    assert all(row["instruction_id"] in mapped for row in trace["events"])
    assert all(row["lowered_wat_offset"] >= 0 for row in mapped.values())
    assert len(trace["coverage"]) < len(mapped)
    assert trace_program(source, 0, mode="native")["status"] == "unsupported"
    with pytest.raises(ValueError, match="unsupported"):
        extract_token_coverage(source, 10)


def test_event_cap_is_an_explicit_failure():
    result = trace_program("i64.const 0 i64.const 0 i64.const 0 i64.const 0", 0, max_events=1)
    assert result["status"] == "failed" and result["outcome"] == "execution_limit"
    assert len(result["events"]) == 1


def test_instruction_sites_compile_through_the_production_lowerer():
    import wasmtime
    from oeis_learn.sandbox.wat_ast import parse
    from oeis_learn.sandbox.lowering import lower_with_map

    source = (
        "i32.const 1 ( if ( result i64 i64 i64 i64 ) ( then i256.const 3 ) ( else i256.const 7 ) )"
    )
    lowered, mapping, _ = lower_with_map(parse(source))
    with wasmtime.Engine() as engine:
        with wasmtime.Module(engine, lowered):
            assert len(mapping) == 4
