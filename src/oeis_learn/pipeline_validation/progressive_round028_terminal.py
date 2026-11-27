"""Publish the unfunded comparison after private replay, preserving the original audit bytes."""

import io
import os
import tempfile
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import _atomic_write, load, reference, seal
from oeis_learn.generalization.progressive.program_search_outcome import (
    build_outcome,
    verify_outcome,
    workstation_receipt,
)
from oeis_learn.generalization.progressive.program_search_report import verify_archive
from oeis_learn.generalization.progressive.program_search_timing import LEARNERS

from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require


def immutable_bytes(path, content, *, verify_only):
    if path.exists() or verify_only:
        require(path.read_bytes() == content, "retained terminal publication differs")
    else:
        _atomic_write(path, content)


def publish_terminal(folder, store, timing_path, timing_receipt, replay, *, verify_only=False):
    from .progressive_round028 import ARCHIVE, SAVED_AUDIT, audit_archive

    result = build_outcome(timing_path)
    public = verify_outcome(io.BytesIO(result))
    prior = audit_archive(folder)
    audit = verify_archive(prior)
    require(
        seal(replay) == replay
        and replay.get("version") == "round028-private-spending-replay/v1"
        and replay.get("scope") == "original_weights_choices_and_scorer_clocks"
        and replay.get("portfolios") == public["measurement"]["portfolios"]
        and replay.get("registration") == timing_receipt["registration"]
        and replay.get("checkpoint_closures") == list(LEARNERS)
        and public["timing"] == timing_receipt["evidence"] == replay["timing"]
        and public["audit"] == store.state["publication"] == audit["report"] == replay["audit"]
        and audit["private_cut"] == replay["private_cut"],
        "terminal audit or private timing parent differs",
    )
    share = confined(folder, ARCHIVE, exists=False)
    saved = confined(folder, SAVED_AUDIT, exists=False)
    original = prior.read_bytes()
    prior_state = store.state.get("terminal")
    if verify_only:
        require(prior_state is not None, "terminal report is awaiting its run boundary")
    # The older audit was originally published at the canonical path. Only promote
    # that exact predecessor. Keep its original ZIP byte-for-byte for all audit readers.
    immutable_bytes(saved, original, verify_only=verify_only)
    area = confined(folder, "private/terminal", exists=False)
    if not verify_only:
        area.mkdir(exist_ok=True, mode=0o700)
    staged = confined(area, "round028.zip", exists=False)
    immutable_bytes(staged, result, verify_only=verify_only)
    closure_path = confined(area, "closure.json", exists=False)
    closure = seal(
        {
            "version": "round028-terminal-boundary/v1",
            "replay": replay,
            "audit": reference(folder, saved),
            "timing": reference(folder, timing_path),
            "staged": reference(folder, staged),
            "report": public["report"],
        }
    )
    if verify_only:
        require(load(closure_path) == closure, "private terminal replay differs")
    else:
        retain(closure_path, closure)
    commitment = {
        "report": public["report"],
        "closure": reference(folder, closure_path),
        "evidence_verdict": public["evidence_verdict"],
        "reason": public["reason"],
        "next_action": public["next_action"],
        "remaining_authorized_gpu_us": 0,
        "pilot_us": public["funding"]["pilot_us"],
        "shift_us": public["funding"]["shift_us"],
        "deficit_us": public["funding"]["deficit_us"],
    }
    require(prior_state is None or prior_state == commitment, "committed terminal boundary differs")

    # Atomically install the staged bytes. On either side of this rename a retry
    # can prove the predecessor or successor without granting any scientific work.
    if share.exists() and share.stat().st_size not in (len(original), len(result)):
        raise ValueError("canonical share differs from the expected audit or final publication")
    current = share.read_bytes() if share.exists() else None
    if current != result:
        require(
            not verify_only and (current == original or current is None),
            "canonical share differs from its registered predecessor or successor",
        )
        with tempfile.TemporaryDirectory(dir=share.parent) as directory:
            incoming = Path(directory) / share.name
            _atomic_write(incoming, result)
            os.replace(incoming, share)
            fd = os.open(share.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    require(verify_outcome(share) == public, "installed public archive differs from private replay")
    if prior_state is None:
        store.append(dict(store.state, status="COMPLETE", terminal=commitment))
    return {
        "mechanical_status": "COMPLETE",
        "scientific_status": "PILOT_UNMEASURED",
        "evidence_verdict": public["evidence_verdict"],
        "next_action": public["next_action"],
        "reason": public["reason"],
        "archive": ARCHIVE,
        "receipt": public,
        "private_replay": "VERIFIED",
        "workstation": workstation_receipt(public, replay["identity"], store.tip),
        "private_scope": "registered_weights_choices_and_original_costs",
        "gpu_dispatches": 0,
        "remaining_authorized_gpu_us": 0,
    }


def close_timing(root, folder, source, store, observed, *, verify_only=False):
    from .progressive_round028_timing import SHARE, replay_spending_inputs

    if observed["reason"] != "REPORT_PILOT_COST_DEFICIT":
        return {
            "mechanical_status": "BLOCKED",
            "scientific_status": "PILOT_UNMEASURED",
            "reason": observed["reason"],
            "gpu_dispatches": 0,
        }, 2
    require(
        observed["archive"] == SHARE + "/round028-timing.zip", "unexpected saved timing publication"
    )
    replay = replay_spending_inputs(root, folder, source, store, observed["receipt"])
    raw = publish_terminal(
        folder,
        store,
        confined(folder, observed["archive"]),
        observed["receipt"],
        replay,
        verify_only=verify_only,
    )
    return raw, 0
