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
        return GroundingResult(
            "unknown", source, (), (0,), "sha256:" + "0" * 64, "not_run", "not_run", 1000, 1.0
        )

    monkeypatch.setattr(api, "ground", fake)
    skeleton = api.parse_ast_placeholders("i256.const_?")
    assert not api.solve_constants(skeleton, [7]).is_sat
    assert api.resolve_program_constants(skeleton.raw_wat, [7])[2] == "UNKNOWN"
    assert len(calls) == 2


def oversize():
    return b"x" * (2 << 20)


def memory_exhaustion():
    return bytearray(2 << 30)


def test_result_and_memory_limits_and_recovery():
    before = {p.pid for p in mp.active_children()}
    assert run_contained(oversize)[0] == "error"
    assert run_contained(memory_exhaustion)[0] == "error"
    assert run_contained(alive) == ("ok", 7)
    assert {p.pid for p in mp.active_children()} == before


def test_partial_frame_is_deadline_bounded():
    import socket, struct, time
    from oeis_learn.sandbox.contained import _receive

    parent, child = socket.socketpair()
    try:
        child.sendall(struct.pack("!I", 100) + b"x")
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            _receive(parent, started + 0.05)
        assert time.monotonic() - started < 1
    finally:
        parent.close()
        child.close()


def test_qualified_activation_rejects_missing_gate_before_dispatch():
    from oeis_learn.experiments.config import activate_qualified_contract
    from oeis_learn.experiments.qualified_models import seal_record, MODES
    from oeis_learn.experiments.artifacts import compute_canonical_digest

    base = {"schema_version": "foundation/v1", "track": "strict_generic"}
    digest = compute_canonical_digest(base)
    contract = seal_record(
        "qualified_contract",
        base_contract_sha256=digest,
        profiles={
            k: digest
            for k in (
                "language",
                "runtime",
                "codec",
                "model",
                "objective",
                "sampling",
                "search",
                "admission",
            )
        },
        gates={"G7": None, "G8": None, "G9": None},
        track="strict_generic",
        initialization="random",
        parent_chain=[],
        objective="sft",
        modes={m: {"enabled": False, "profile_sha256": None} for m in MODES},
    )
    with pytest.raises(ValueError, match="G7"):
        activate_qualified_contract(
            contract,
            g7_report={"gate": "G7", "status": "PASS"},
            evidence_root=".",
            resolve=lambda _: base,
        )
