"""Reuse an authenticated development runtime and rebuild only the current application wheel."""

import hashlib
import json
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import zipfile

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    physical_root_id,
    validate_grant,
)
from .identity import confined
from .progressive_research_inputs import checked_file
from .progressive_round028_receipts import verify_role_mounts
from .progressive_velocity_qualification import _receipt, qualification_parent

BOOTSTRAP = "src/oeis_learn/pipeline_validation/progressive_velocity_wheel.py"
INSTALLED = "oeis_learn/experiments/qualification-sources.json"
# The root Dockerfile owns the ROCm base, native tools and Python environment.
# Check their complete tracked inputs in addition to reopening the installed bytes.
ENVIRONMENT = (
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    "rustfmt.toml",
    "docker/dev",
    "crates",
    "src/oeis_learn/synthesis/records.schema.json",
    "src/oeis_learn/synthesis/engine-policy.json",
    "src/oeis_learn/sandbox/preamble.wat",
)


def compatible(root, original, revision):
    if not all(
        isinstance(r, str) and re.fullmatch("[0-9a-f]{40}", r) for r in (original, revision)
    ):
        raise ValueError("two full commit identities required for runtime reuse")
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", original, revision, "--", *ENVIRONMENT]
    )
    if result.returncode not in (0, 1):
        raise ValueError("cannot verify the original runtime build inputs")
    return result.returncode == 0


def learner_runtime(root, previous):
    """Reopen the settled learner rehearsal without depending on the latest session status."""
    area = confined(root, f"run-state/discovery/{SESSION}/qualification/{previous}")
    binding, grant, receipt, req = [
        load(confined(area, name + ".json")) for name in ("binding", "grant", "receipt", "request")
    ]
    if (
        binding != seal(binding)
        or binding["version"] != "round029-learner-rehearsal/v1"
        or binding["grant"] != grant["identity"]
        or previous != grant["identity"][7:]
    ):
        raise ValueError("runtime does not belong to a settled learner rehearsal")
    approval = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant,
        approval,
        root=physical_root_id(root),
        action="qualify",
        parent=qualification_parent(
            binding["candidate"], children=binding.get("child_restore", False)
        ),
    )
    journal = BudgetJournal.inspect_existing(
        confined(area, "accounting"), grant["identity"], grant["resources"]
    )
    if receipt != _receipt(area, confined(area, receipt["attempt"]), binding) or (
        journal["pending"]
        or not receipt["debits"]
        or not set(receipt["debits"]) <= set(journal["committed"])
    ):
        raise ValueError("registered runtime image lacks its original verified debit")
    return seal(
        dict(
            grant=grant["identity"],
            receipt=receipt["identity"],
            candidate=binding["candidate"],
            source=req["source"],
            image=req["image"],
        )
    )


def cached_runtime(root, folder, revision, engine):
    for path in sorted(folder.glob("qualification/*/receipt.json")):
        record = load(confined(root, path.parent / "binding.json"))
        if record.get("version") != "round029-learner-rehearsal/v1":
            continue
        if not compatible(root, record["candidate"], revision):
            continue
        parent = learner_runtime(root, path.parent.name)
        try:
            image = engine.image(parent["image"])
        except subprocess.CalledProcessError:
            continue
        if image != parent["image"]:
            raise ValueError("qualified image resolved to different bytes")
        return parent
    return None


def package_worker(root, source, area, out, revision, parent, source_hash, host, stage, seconds):
    archive = out / "candidate.tar.gz"
    with archive.open("xb") as body:
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar.gz", revision],
            stdout=body,
            check=True,
            timeout=seconds(),
        )
    bootstrap = out / "candidate-wheel.py"
    shutil.copyfile(confined(source, BOOTSTRAP), bootstrap)
    source_ref = reference(area, archive)
    target = out / "package"
    target.mkdir()
    stage(
        host,
        source,
        parent["image"],
        target,
        [
            "python",
            "-I",
            "-B",
            "/work/inputs/candidate-wheel.py",
            "package",
            "/work/inputs/candidate.tar.gz",
            source_ref["sha256"],
            parent["source"],
            source_hash,
        ],
        gpu=False,
        seconds=seconds(),
        readonly_inputs=[
            (bootstrap, "/work/inputs/candidate-wheel.py"),
            (archive, "/work/inputs/candidate.tar.gz"),
        ],
    )
    (wheel,) = target.glob("oeis_learn-*-py3-none-any.whl")
    return seal(
        dict(
            parent=parent,
            archive=source_ref,
            bootstrap=reference(area, bootstrap),
            wheel=reference(area, wheel),
        )
    )


def wheel_sources(wheel, source):
    """Check every application byte before the candidate wheel can be executed."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        snapshot = json.loads(archive.read(INSTALLED))
        if digest(snapshot) != source:
            raise ValueError("wheel manifest differs from the registered candidate")
        expected = {p[4:]: h for p, h in snapshot.items() if p.startswith("src/oeis_learn/")}
        actual = {p for p in names if p.startswith("oeis_learn/") and p != INSTALLED}
        metadata = wheel.name.split("-py3-none-any.whl")[0] + ".dist-info"
        if (
            len(set(names)) != len(names)
            or actual != expected.keys()
            or any(
                p not in expected
                and p != INSTALLED
                and (PurePosixPath(p).parts[0] != metadata or ".." in PurePosixPath(p).parts)
                for p in names
            )
        ):
            raise ValueError("candidate wheel inventory differs")
        for name, sha in expected.items():
            with archive.open(name) as content:
                if "sha256:" + hashlib.file_digest(content, "sha256").hexdigest() != sha:
                    raise ValueError("candidate wheel member differs: " + name)
    return snapshot


def package_evidence(root, area, runtime, binding):
    if runtime != seal(runtime):
        raise ValueError("candidate wheel acquisition identity differs")
    parent = runtime["parent"]
    if parent != learner_runtime(root, parent["grant"][7:]) or not compatible(
        root, parent["candidate"], binding["candidate"]
    ):
        raise ValueError("original qualified runtime or environment inputs differ")
    archive, bootstrap, wheel = [
        checked_file(area, runtime[k]) for k in ("archive", "bootstrap", "wheel")
    ]
    target = archive.parent / "package"
    if wheel.parent != target or bootstrap != archive.parent / "candidate-wheel.py":
        raise ValueError("candidate wheel belongs to another packaging attempt")
    snapshot = wheel_sources(wheel, binding["source"])
    if snapshot[BOOTSTRAP] != runtime["bootstrap"]["sha256"]:
        raise ValueError("wheel or bootstrap differs from the registered application source")
    inspection = load(confined(target, "container-inspect.json"))
    verify_role_mounts(
        inspection,
        parent["image"],
        {
            "/work/inputs/candidate-wheel.py": (bootstrap, False),
            "/work/inputs/candidate.tar.gz": (archive, False),
            "/work/output": (target, True),
        },
    )
    limits = inspection[0]["HostConfig"]
    if limits.get("Devices") or limits.get("DeviceRequests"):
        raise ValueError("candidate packaging must run without accelerator devices")
    proc, clean = [load(confined(target, p)) for p in ("process.json", "cleanup.json")]
    if (
        proc["outcome"] != "COMPLETE"
        or proc["returncode"] != 0
        or proc["cleanup_failure"]
        or (clean["acknowledged"] is not True or clean["within_bound"] is not True)
    ):
        raise ValueError("candidate wheel installation or cleanup failed")


def wheel_mounts(area, runtime):
    wheel = checked_file(area, runtime["wheel"])
    return [
        (checked_file(area, runtime["bootstrap"]), "/work/inputs/candidate-wheel.py"),
        (wheel, "/work/inputs/" + wheel.name),
    ]
