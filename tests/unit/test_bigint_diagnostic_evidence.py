import time
from types import SimpleNamespace

import pytest

from oeis_learn.fresh import bigint_diagnostics as diagnostics
from oeis_learn.fresh.synthesis_policy import select_visible
from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.contracts import ResourceProfile


def selection(source="(program (return $index))"):
    prompt = VisiblePrompt(tuple(map(str, range(20))), tuple(map(str, range(20))))
    chosen = select_visible(
        [
            dict(
                source=source,
                source_sha256=source_identity(source),
                verified=True,
                visible_count=20,
                cost=0,
            )
        ]
    )
    return dict(
        selected=chosen, prompt_sha256=prompt.sha256, verification=dict(prompt_sha256=prompt.sha256)
    )


def truth():
    return {"prompt": dict(indices=list(map(str, range(100))), values=list(map(str, range(100))))}


class Client:
    limits = ResourceProfile()

    def execute(self, source, indices, **kwargs):
        return dict(outcome="ok", values=list(indices))


def test_expired_score_cannot_inherit_generation_completeness():
    result = diagnostics.score(
        {"prompt": dict(selections={"rescue": selection()}, complete=True)},
        truth(),
        Client(),
        deadline=time.monotonic() - 1,
    )
    outcome = result["outcomes"]["prompt"]["rescue"]
    assert not outcome["complete"] and not outcome["success"]
    assert outcome["scoring"]["reason"] == "deadline"


def test_no_selection_uses_its_own_arm_evidence():
    result = diagnostics.score(
        {
            "prompt": dict(
                selections={},
                complete=False,
                arm_evidence={
                    "rescue": dict(complete=True, reason="unsupported"),
                    "absent": dict(complete=False, reason="generation_incomplete"),
                },
            )
        },
        truth(),
        Client(),
        deadline=time.monotonic() + 10,
    )
    assert result["outcomes"]["prompt"]["rescue"]["complete"]
    assert not result["outcomes"]["prompt"]["absent"]["complete"]
    assert not result["outcomes"]["prompt"]["rescue"]["success"]


def test_legacy_missing_selection_is_explicitly_unattributed():
    result = diagnostics.score(
        {"prompt": dict(selections={}, complete=True)},
        truth(),
        Client(),
        deadline=time.monotonic() + 10,
    )
    outcome = result["outcomes"]["prompt"]["rescue"]
    assert not outcome["complete"]
    assert outcome["reason"] == "legacy_arm_evidence_missing"


def test_selected_candidate_with_complete_independent_horizon_is_successful():
    result = diagnostics.score(
        {"prompt": dict(selections={"rescue": selection()}, complete=False)},
        truth(),
        Client(),
        deadline=time.monotonic() + 10,
    )
    outcome = result["outcomes"]["prompt"]["rescue"]
    assert outcome["complete"] and outcome["success"]
    assert outcome["scoring"]["production"]["coverage"] == 100
    assert outcome["scoring"]["reference"]["coverage"] == 100


@pytest.mark.parametrize(
    "production",
    [
        dict(outcome="ok", values=["0"]),
        dict(outcome="resource_exhausted", reason="deadline", values=[]),
    ],
)
def test_partial_or_failed_execution_is_incomplete(production):
    client = SimpleNamespace(limits=ResourceProfile(), execute=lambda *a, **kw: production)
    result = diagnostics.score(
        {"prompt": dict(selections={"rescue": selection()}, complete=True)},
        truth(),
        client,
        deadline=time.monotonic() + 10,
    )
    assert not result["outcomes"]["prompt"]["rescue"]["complete"]


def test_rescue_coverage_does_not_include_later_stochastic_fits(monkeypatch):
    from oeis_learn.decoder.bigint_codec import decode_body, encode_body
    from oeis_learn.fresh.bigint_config import defaults

    tokens = encode_body("(program (return $index))")
    source = decode_body(tokens)
    monkeypatch.setattr(
        diagnostics,
        "_raw",
        lambda *a, **kw: (None, dict(complete=True, outcome="visible_mismatch", reason=None)),
    )
    fits = iter(["unsupported", "resource_exhausted"])
    client = SimpleNamespace(fit=lambda *a, **kw: dict(outcome=next(fits), admitted=[]))
    row = dict(id="prompt", indices=list(map(str, range(20))), values=["0"] * 20)
    portfolio = diagnostics._portfolio(
        None, None, row, defaults(), client, deadline=time.monotonic() + 60, attempt_count=2
    )
    next(portfolio)
    portfolio.send(
        dict(
            source=source,
            tokens=tokens,
            complete=True,
            generation_seconds=0,
            generation_batch_size=1,
        )
    )
    with pytest.raises(StopIteration) as stopped:
        portfolio.send(
            dict(
                source=source,
                tokens=tokens,
                complete=True,
                generation_seconds=0,
                generation_batch_size=1,
            )
        )
    evidence = stopped.value.value["arm_evidence"]
    assert evidence["rescue"]["complete"]
    assert not evidence["equal_budget"]["complete"]
    assert evidence["rescue"]["fitting"]["outcomes"] == ["unsupported"]


def test_compact_all_prompt_evidence_survives_example_cap_and_pack(tmp_path):
    import json
    import zipfile
    from oeis_learn.fresh.bigint_worker import write_diagnostic_evidence
    from oeis_learn.fresh.bigint_bundle import pack, CORE
    from oeis_learn.fresh.common import read, write

    rows, records = [], {}
    for n in range(32):
        key = str(n)
        rows.append(
            dict(
                id=key,
                indices=list(map(str, range(20))),
                values=["0"] * 20,
                hidden_values=["NEVER_EXPORT"],
            )
        )
        attempt = dict(
            position=0,
            seed=n,
            source_sha256="source",
            source="(program (return $index))" + " " * 60000,
            generation=dict(complete=True, outcome="decoded", reason=None),
            raw=dict(complete=True, outcome="visible_mismatch", reason=None),
            fitting=dict(complete=False, outcome="unknown", reason="timeout"),
            native=dict(private_debug="unbounded trace is excluded"),
            tokens=[1, 2, 3],
        )
        records[key] = dict(
            arm_evidence=dict(rescue=dict(complete=False, reason="timeout")),
            details=dict(correct=dict(details=[attempt])),
        )
    role = tmp_path / "role"
    write_diagnostic_evidence(role, rows, records)
    saved = {}
    report = tmp_path / "report"
    for path in (role / "evidence").glob("*.json"):
        saved.update(read(path))
        write(report / "details" / ("000-evidence-" + path.name), read(path))
    assert set(saved) == {str(i) for i in range(32)}
    assert saved["31"]["attempts"]["correct"][0]["fitting"]["reason"] == "timeout"
    assert "native" not in saved["31"]["attempts"]["correct"][0]
    for name in CORE:
        write(report / name, dict(final_access="sealed"))
    # Verbose examples sort before "000-evidence" lexically and would crowd
    # out the later evidence chunk without the packer's explicit priority.
    for n in range(16):
        write(report / f"details/0-tiny-example-{n:02}.json", dict(trace="x" * ((1 << 20) - 100)))
    target = tmp_path / "report.zip"
    pack(report, target, max_bytes=16 << 20)
    with zipfile.ZipFile(target) as bundle:
        evidence_files = [
            name for name in bundle.namelist() if name.startswith("details/000-evidence")
        ]
        assert evidence_files
        assert "NEVER_EXPORT" not in "".join(bundle.read(n).decode() for n in evidence_files)
        omissions = json.loads(bundle.read("bundle-manifest.json"))["omitted"]
        assert omissions and all("0-tiny-example-" in row["path"] for row in omissions)
        recovered = {}
        for name in evidence_files:
            recovered.update(json.loads(bundle.read(name)))
        assert set(recovered) == set(saved)


def test_fit_preserves_independent_admission_exhaustion_reason(monkeypatch):
    from oeis_learn.synthesis import client as clients
    from oeis_learn.synthesis.admission import Admission

    source = "(program (return $index))"
    monkeypatch.setattr(
        clients, "admit", lambda *a, **kw: Admission("resource_exhausted", "reference_limit")
    )
    with clients.SynthesisClient(
        "sha256:" + "1" * 64, worker=SimpleNamespace(close=lambda: None)
    ) as client:
        client.runtime_identity = lambda profile: "mock-runtime"
        client.request = lambda *a, **kw: dict(
            results=[
                dict(
                    outcome="assignment_found",
                    assignments=[
                        dict(
                            original_source_sha256=source_identity(source),
                            source_sha256=source_identity(source),
                        )
                    ],
                )
            ]
        )
        result = client.fit(
            source, VisiblePrompt(tuple(map(str, range(20))), tuple(map(str, range(20))))
        )
    assert result["outcome"] == "resource_exhausted" and not result["admitted"]
    assert result["reason"] == "reference_limit"
    assert result["admission"][0]["outcome"] == "resource_exhausted"


@pytest.mark.parametrize(
    "outcome,reason",
    [
        ("resource_exhausted", "deadline"),
        ("resource_exhausted", "worker_rss"),
        ("verification_failure", "stale or mismatched response"),
    ],
)
def test_empty_worker_response_preserves_failure_class_and_reason(outcome, reason):
    from oeis_learn.synthesis.client import SynthesisClient

    worker = SimpleNamespace(generation=1, close=lambda: None)
    worker.call = lambda request, **kw: dict(
        results=[],
        unstarted=[row["candidate_id"] for row in request["candidates"]],
        complete=False,
        outcome=outcome,
        reason=reason,
        generation=2,
        metrics=dict(rss_peak_bytes=123),
    )
    source = "(program (return $index))"
    with SynthesisClient("sha256:" + "1" * 64, worker=worker) as client:
        client.runtime_identity = lambda profile: "mock-runtime"
        fit = client.fit(
            source, VisiblePrompt(tuple(map(str, range(20))), tuple(map(str, range(20))))
        )
        executed = client.execute(source, ["0"])
    assert fit["outcome"] == executed["outcome"] == outcome
    assert fit["reason"] == executed["reason"] == reason
    assert fit["native"]["transport"][0]["generation"] == 2
