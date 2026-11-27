"""Disposable installed release checks. Nothing here dispatches research."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    read_bound_record,
    reference,
    seal,
)
from .progressive_research_inputs import checked_file

HISTORICAL = dict(
    candidate_score_bytes=209_010_758,
    campaign_bytes=56_568_256,
    calibration_bytes=49_000_000,
    tasks=352,
    journal_events=290_000,
)


def scale_targets(measured, tasks):
    return {
        k: max(floor, tasks if k == "tasks" else measured.get(k, 0))
        for k, floor in HISTORICAL.items()
    }


def _us(start):
    return (time.monotonic_ns() - start) // 1000


def _corpus(folder, size, tasks, varied):
    folder.mkdir()
    generator = random.Random(28)
    artifacts = []
    compressed = folder / "transport.zip"
    began = time.monotonic_ns()
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i in range(tasks):
            path = folder / f"batch-{i}.bin"
            remaining = size // tasks + int(i < size % tasks)
            with path.open("xb") as stream:
                while remaining:
                    count = min(remaining, 1 << 20)
                    stream.write(generator.randbytes(count) if varied else bytes(count))
                    remaining -= count
            artifacts.append(reference(folder, path))
            archive.write(path, path.name)
    manifest = folder / "manifest.json"
    atomic_write(manifest, seal(dict(artifacts=artifacts)))
    ref = reference(folder, manifest)
    cold = _us(began)
    began = time.monotonic_ns()
    reopened = read_bound_record(folder, ref)
    with zipfile.ZipFile(compressed) as archive:
        if archive.namelist() != [r["path"] for r in reopened["artifacts"]]:
            raise ValueError("release archive batch order differs")
        for r in reopened["artifacts"]:
            checked_file(folder, r)
            with archive.open(r["path"]) as member:
                checksum = hashlib.file_digest(member, "sha256").hexdigest()
            if (
                r["sha256"] != "sha256:" + checksum
                or archive.getinfo(r["path"]).file_size != r["bytes"]
            ):
                raise ValueError("standalone expanded batch differs")
    return dict(
        expanded_bytes=sum(r["bytes"] for r in artifacts),
        compressed_bytes=compressed.stat().st_size,
        files=len(artifacts),
        cold_us=cold,
        warm_us=_us(began),
    )


def _journal(folder, events):
    if events <= 0 or events % 2:
        raise ValueError("paired journal events required")
    caps = dict(dict.fromkeys(KEYS, 0), elapsed_us=events // 2)
    journal = BudgetJournal(
        folder, digest("release scale fixture"), caps, journal_bytes=events * 1024 + (1 << 20)
    )
    previous = digest(journal.binding)
    cost = dict(dict.fromkeys(KEYS, 0), elapsed_us=1)
    # Generate the registered historical-size input in linear time, then exercise the
    # production cold replay and warm cache including every reservation and settlement.
    with journal.path.open("x") as out:
        for i in range(events):
            first = i % 2 == 0
            event = seal(
                dict(
                    event="RESERVE" if first else "COMMIT",
                    index=i,
                    operation=f"scale/elapsed/{i//2}",
                    boot="release-fixture",
                    start_us=i // 2,
                    end_us=None if first else i // 2 + 1,
                    previous=previous,
                    cost=cost,
                )
            )
            out.write(json.dumps(event, separators=(",", ":")) + "\n")
            previous = event["identity"]
    began = time.monotonic_ns()
    first = journal.snapshot()
    cold = _us(began)
    began = time.monotonic_ns()
    second = journal.snapshot()
    warm = _us(began)
    if (
        first != second
        or first["pending"]
        or first["consumed"] != caps
        or first["watermark"] != previous
        or len(first["committed"]) != events // 2
    ):
        raise ValueError("original-scale journal replay differs")
    return dict(
        events=events,
        committed=events // 2,
        bytes=journal.path.stat().st_size,
        cold_us=cold,
        warm_us=warm,
    )


def exercise_scale(output, targets):
    with tempfile.TemporaryDirectory(dir=output) as temp:
        folder = Path(temp)
        sizes = {}
        for name in ("campaign_bytes", "calibration_bytes"):
            path = folder / (name + ".json")
            record = seal(
                dict(
                    fixture="registered-release-scale",
                    slots=list(range(targets["tasks"])),
                    payload="x" * targets[name],
                )
            )
            atomic_write(path, record, compact=True)
            if (
                read_bound_record(folder, reference(folder, path), identity=record["identity"])
                != record
            ):
                raise ValueError("large campaign or calibration round trip differs")
            sizes[name] = path.stat().st_size
            del record
        corpora = {
            name: _corpus(folder / name, targets["candidate_score_bytes"], targets["tasks"], varied)
            for name, varied in (("compressible", False), ("varied", True))
        }
        ledger = _journal(folder / "journal", targets["journal_events"])
        # Both encodings, their uncompressed producers, and the journal coexist at this boundary.
        stats = [p.stat() for p in folder.rglob("*")]
        return dict(
            targets=targets,
            corpora=corpora,
            journal=ledger,
            **sizes,
            peak_allocated_bytes=sum(s.st_blocks * 512 for s in stats),
            peak_inodes=len(stats),
        )


def release_cases():
    # Frozen v1 membership is also used by later archive-only readers.
    return [
        "tests/contract/test_round028_routing.py",
        "tests/integration/test_round028_audit.py",
        "tests/integration/test_round028_rehearsal.py",
        "tests/integration/test_round028_receipts.py",
        "tests/integration/test_round028_panels.py",
        "tests/integration/test_round028_timing.py",
        "tests/integration/test_round028_terminal.py",
        "tests/integration/test_round028_pilot.py",
        "tests/integration/test_round028_pilot_resume.py",
        "tests/integration/test_round028_branches.py",
        "tests/integration/test_round028_convergence.py",
        "tests/integration/test_round028_release.py",
        "tests/unit/test_round028_training.py",
        "tests/unit/test_progressive_storage_plan.py",
        "tests/integration/test_progressive_reporting.py",
        "tests/integration/test_progressive_handoff.py",
        "tests/contract/test_progressive_evaluation.py",
        "tests/contract/test_progressive_packaging.py",
        "tests/integration/test_progressive_lifecycle.py",
        "tests/integration/test_progressive_evaluation.py",
        "tests/unit/test_progressive_recovery_transactions.py",
        "tests/unit/test_progressive_decisions.py",
        "tests/unit/test_progressive_freeze_progress.py",
        "tests/unit/test_progressive_trials.py",
        "tests/unit/test_progressive_trial_eligibility.py",
        "tests/integration/test_progressive_recovery_updates.py",
        "tests/unit/test_foundation_local.py::test_two_candidate_sources_contend_for_donor_gpu_lease",
    ]


def installed_suite(output, mode):
    if mode not in ("source", "wheel"):
        raise ValueError("explicit source or wheel suite required")
    source = Path("/opt/oeis-build")
    cases = release_cases()
    with tempfile.TemporaryDirectory(dir=output) as temp:
        scratch = Path(temp)
        search = [str(source)] if mode == "wheel" else [str(source / "src"), str(source)]
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(search),
            TMPDIR=temp,
            OEIS_REQUIRE_BIGINT_BUILD="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="2",
        )
        origin = subprocess.check_output(
            [sys.executable, "-c", "import oeis_learn; print(oeis_learn.__file__)"],
            cwd=source,
            env=env,
            text=True,
        ).strip()
        if Path(origin).is_relative_to(source / "src") != (mode == "source"):
            raise ValueError("release suite imported the wrong package")
        report = output / (mode + ".xml")
        began = time.monotonic_ns()
        with (output / (mode + ".log")).open("wb") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-o",
                    "tmp_path_retention_policy=failed",
                    "--basetemp",
                    str(scratch / "pytest"),
                    "--junitxml",
                    str(report),
                    *(str(source / p) for p in cases),
                ],
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        results = ET.parse(report).findall(".//testcase")
        if (
            proc.returncode
            or not results
            or any(r.find(k) is not None for r in results for k in ("failure", "error", "skipped"))
        ):
            raise ValueError(f"{mode} release suite incomplete; inspect {mode}.xml")
        return dict(
            mode=mode,
            files=len(cases),
            roster=digest(cases),
            tests=len(results),
            elapsed_us=_us(began),
            junit=reference(output, report),
        )
