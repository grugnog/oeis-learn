"""The budget decision closes the run from verified old evidence and survives publication interruptions."""

import io
import json
import runpy
import subprocess
import zipfile
from pathlib import Path

import pytest

from oeis_learn.generalization.progressive import program_search_panel_report as panels
from oeis_learn.generalization.progressive import program_search_timing as timing
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.program_search_outcome import (
    MEMBERS,
    archive_bytes,
    build_outcome,
    verify_outcome,
    workstation_receipt,
)
from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
    encode,
    publish,
    verify_transport,
)
from oeis_learn.pipeline_validation.progressive_round028 import ARCHIVE
from oeis_learn.pipeline_validation.progressive_round028_receipts import recover
from tests.integration.test_round028_panels import historical_clocks, summary
from tests.integration.test_round028_receipts import REVISION, SOURCE
from tests.integration.test_round028_receipts import original as original  # noqa: PLC0414 - shared fixture
from tests.integration.test_round028_timing import measured


@pytest.fixture
def collection(original, tmp_path, monkeypatch):
    o = original
    old = recover(o.workspace, o.folder, SOURCE, REVISION, o.store)
    rehearsal = tmp_path / "rehearsal.zip"
    trainer = publish(rehearsal, old, o.folder / ARCHIVE)
    panel = panels.make_report(
        seal({"candidate": "c" * 40}),
        trainer,
        summary(deficit=False),
        historical_clocks(),
        timings={"elapsed_us": 10, "role_us": 5, "unknown_role_intervals": 0, "image_build_us": 1},
    )
    parent = tmp_path / "panels.zip"
    panels.publish(parent, panel, rehearsal)
    monkeypatch.setattr(timing, "PANEL", panel["identity"])

    def build(name, *, expensive=True, incomplete=False):
        blocks = measured()
        for block in blocks.values():
            if expensive:
                # Permanent case clocks still close. Expensive cold roles exhaust the registered shift.
                block["gpu_wall_us"] += 5_000_000_000
            if incomplete:
                block["cells"]["oeis"]["closed"] = 1
        value = seal(
            {
                "version": timing.VERSION,
                "panel": panel["identity"],
                "candidate": "d" * 40,
                "registration": digest("timing-registration"),
                "scope": "previously_exposed_timing_only",
                "blocks": blocks,
                "forecast": timing.project(blocks, timing.parent_plan(panel, old)),
                "image_build_us": {"hip": 2, "cpu": 1},
                "space_before": {"free_bytes": 10, "free_inodes": 5},
                "space_after": {"free_bytes": 8, "free_inodes": 4},
            }
        )
        path = tmp_path / (name + ".zip")
        panels.publish_child(path, value, parent, timing.MEMBERS, timing.verify_timing)
        return path

    return o, build


def test_verified_deficit_is_a_terminal_resource_result_with_no_formal_gain(collection, tmp_path):
    _, build = collection
    source = build("expensive")
    result = build_outcome(source)
    assert result == build_outcome(source)
    # Earlier ZIP readers allow unused trailing bytes. Only the reviewed JSON
    # projections may reach the terminal share, including when copied from such an archive.
    with source.open("ab") as stream:
        stream.write(b"private-sequence-marker private-program-marker private-trace-marker")
    assert build_outcome(source) == result
    assert b"private-sequence-marker" not in result and b"private-program-marker" not in result
    path = tmp_path / "round028.zip"
    path.write_bytes(result)
    receipt = verify_transport(path)
    assert receipt == verify_outcome(path)
    assert (
        receipt["evidence_verdict"] == "INCOMPLETE" and receipt["next_action"] == "RETAIN_BASELINE"
    )
    assert receipt["remaining_authorized_gpu_us"] == 0 and receipt["funding"]["deficit_us"] > 0
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == list(MEMBERS)
        value = json.loads(z.read("outcome.json"))
        assert (
            value["comparison"]["primary_gain"] is None and value["comparison"]["new_4m_runs"] == 0
        )
        assert (
            value["measurement"]["portfolios"] == 40
            and value["comparison"]["assessment"] == "SEALED"
        )
    for path in (build("affordable", expensive=False), build("partial", incomplete=True)):
        with pytest.raises(ValueError, match="cost deficit"):
            build_outcome(path)


def test_archive_recomputes_cost_and_rejects_privacy_or_parent_changes(collection, tmp_path):
    _, build = collection
    with zipfile.ZipFile(io.BytesIO(build_outcome(build("original")))) as z:
        records = {name: z.read(name) for name in z.namelist()}

    def reject(changed, *, trailing=b""):
        path = tmp_path / "altered.zip"
        with zipfile.ZipFile(path, "w") as z:
            for name, raw in changed.items():
                z.writestr(zipfile.ZipInfo(name), raw)
        if trailing:
            with path.open("ab") as out:
                out.write(trailing)
        with pytest.raises(ValueError):
            verify_transport(path)

    for key, val in (
        ("remaining_authorized_gpu_us", 1),
        ("timing", digest("other parent")),
        ("next_action", "CONFIRM"),
        ("secret_terms", ["private-marker"]),
    ):
        target = json.loads(records["outcome.json"])
        target[key] = val
        reject(dict(records, **{"outcome.json": encode(seal(target))}))
    target = json.loads(records["timing.json"])
    target["forecast"]["pilot_us"] -= 1
    reject(dict(records, **{"timing.json": encode(seal(target))}))
    reject(records, trailing=b"private-hidden-trace")
    reject(dict(records, **{"private-program.json": b"private-marker"}))


def test_collection_carries_one_bound_workstation_receipt_and_the_verified_copy(
    collection, monkeypatch
):
    original, build = collection
    canonical = build_outcome(build("original"))
    public = verify_outcome(io.BytesIO(canonical))
    attestation = workstation_receipt(public, digest("private replay"), digest("journal tip"))
    (original.folder / ARCHIVE).write_bytes(canonical)
    tools = original.workspace / "run-state/discovery-tools"
    tools.mkdir()
    calls = []

    def invoke(cmd, *, stdout, **kwargs):
        action = cmd[cmd.index("--action") + 1]
        calls.append(action)
        if action == "finalize":
            output = dict(
                archive=ARCHIVE,
                private_replay="VERIFIED",
                receipt=public,
                workstation=attestation,
                private_trace="private-marker",
            )
        else:
            output = verify_transport(Path(cmd[cmd.index("--archive") + 1]))
        stdout.write(encode(output) + b"\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", invoke)
    script = SOURCE / "specs/015-learned-program-search/validation/collect_audit.py"
    assert (
        runpy.run_path(str(script))["collect"](original.workspace, "d" * 40, "collect-final") == 0
    )
    assert calls == ["finalize", "verify"]
    package = next(tools.glob("round028-final-*/round028-final-workstation.zip"))
    assert verify_transport(package)["workstation"] == attestation
    assert b"private-marker" not in package.read_bytes()
    with zipfile.ZipFile(package) as z:
        assert z.read("round028.zip") == canonical
    for change in (
        {"scope": "pilot_completed"},
        {"report": digest("other report")},
        {"private_replay": "private-marker"},
        {"private_program": "private-marker"},
    ):
        changed = seal(dict(attestation, **change))
        raw = archive_bytes({"workstation.json": encode(changed), "round028.zip": canonical})
        with pytest.raises(ValueError):
            verify_transport(io.BytesIO(raw))


def test_publication_recovers_without_replacing_an_unrelated_archive(collection, monkeypatch):
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.program_search_report import verify_archive
    from oeis_learn.pipeline_validation.progressive_round028 import (
        audit_archive,
        audit_store,
        status,
    )
    from oeis_learn.pipeline_validation.progressive_round028_terminal import publish_terminal

    o, build = collection
    price = o.folder / "share/evaluation-timing/round028-timing.zip"
    price.parent.mkdir(parents=True)
    price.write_bytes(build("measured").read_bytes())
    original_zip = (o.folder / ARCHIVE).read_bytes()
    audit = verify_archive(o.folder / ARCHIVE)
    receipt = timing.verify_timing(price)
    replay = seal(
        {
            "version": "round028-private-spending-replay/v1",
            "timing": receipt["evidence"],
            "audit": audit["report"],
            "private_cut": audit["private_cut"],
            "scope": "original_weights_choices_and_scorer_clocks",
            "portfolios": 40,
            "registration": receipt["registration"],
            "checkpoint_closures": list(timing.LEARNERS),
        }
    )
    # The last durable transition can be interrupted after the official share has been installed.
    old_append = o.store.append
    monkeypatch.setattr(
        o.store, "append", lambda *a: (_ for _ in ()).throw(OSError("interrupted commit"))
    )
    with pytest.raises(OSError, match="interrupted"):
        publish_terminal(o.folder, o.store, price, receipt, replay)
    assert (
        audit_archive(o.folder).read_bytes() == original_zip
        and audit_archive(o.folder) != o.folder / ARCHIVE
    )
    assert verify_outcome(o.folder / ARCHIVE)["evidence_verdict"] == "INCOMPLETE"
    monkeypatch.setattr(o.store, "append", old_append)
    raw = publish_terminal(o.folder, o.store, price, receipt, replay)
    tip = o.store.tip
    assert raw["receipt"] == verify_transport(o.folder / ARCHIVE)
    assert raw["workstation"] == workstation_receipt(raw["receipt"], replay["identity"], tip)
    assert publish_terminal(o.folder, o.store, price, receipt, replay, verify_only=True) == raw
    assert publish_terminal(o.folder, o.store, price, receipt, replay) == raw and o.store.tip == tip
    current = status(o.folder)
    assert (
        current["next_action"] == "RETAIN_BASELINE" and current["remaining_authorized_gpu_us"] == 0
    )
    assert audit_store(o.folder, "e" * 40).state["publication"] == audit["report"]
    (o.folder / ARCHIVE).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_terminal(o.folder, o.store, price, receipt, replay, verify_only=True)
    assert publish_terminal(o.folder, o.store, price, receipt, replay) == raw and o.store.tip == tip
    (o.folder / ARCHIVE).write_bytes(b"unrelated evidence")
    with pytest.raises(ValueError):
        publish_terminal(o.folder, o.store, price, receipt, replay)
    assert (o.folder / ARCHIVE).read_bytes() == b"unrelated evidence"
