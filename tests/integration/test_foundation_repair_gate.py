import multiprocessing as mp
import os
import pytest
from oeis_learn.sandbox.contained import run_contained
from oeis_learn.evaluation.repair_gate import require_g7
from oeis_learn.evaluation.readiness import require_architecture_measurement_gates


def hang():
    while True:
        pass


def crash():
    os._exit(9)


def alive():
    return 7


def test_contained_timeout_crash_and_recovery():
    before = {p.pid for p in mp.active_children()}
    assert run_contained(hang, timeout_ms=100)[0] == "timeout"
    assert run_contained(crash)[0] == "error"
    assert run_contained(alive) == ("ok", 7)
    assert {p.pid for p in mp.active_children()} == before


def test_g7_does_not_accept_partial_or_self_declared_pass():
    with pytest.raises(ValueError):
        require_g7({"gate": "G7", "status": "PARTIAL"})
    with pytest.raises(ValueError):
        require_architecture_measurement_gates({"gate": "G7", "status": "PASS"})


def test_original_apis_share_ground_service(monkeypatch):
    from oeis_learn.decoder import constant_solver as api
    from oeis_learn.decoder.grounding import GroundingResult
    calls = []
    def fake(source, terms, **kwargs):
        calls.append((source, terms))
        return GroundingResult("unknown", source, (), (0,), "sha256:"+"0"*64,
                               "not_run", "not_run", 1000, 1.0)
    monkeypatch.setattr(api, "ground", fake)
    skeleton = api.parse_ast_placeholders("i256.const_?")
    assert not api.solve_constants(skeleton, [7]).is_sat
    assert api.resolve_program_constants(skeleton.raw_wat, [7])[2] == "UNKNOWN"
    assert len(calls) == 2
