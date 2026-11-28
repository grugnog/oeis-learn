"""Read the original zero-cut authority, debits and bounded supervisor evidence."""

import fcntl
import re
import runpy
import subprocess
import sys

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    operation_key,
    physical_root_id,
    validate_grant,
)
from . import progressive_velocity_checkpoint_host as checkpoints
from . import progressive_velocity_prescore as prescore
from .identity import confined
from .supervision import log_tail

PHASES = (
    "freeze_trials",
    "package",
    "verify_paid_package",
    "build_runtime",
    "package_runtime",
    "publish_prescore",
    "measure",
)


def worker_source(root, reader, revision):
    if not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("full original worker commit required")
    from .progressive_round028 import candidate_revision

    current = candidate_revision(reader, root)
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", revision, current], check=True
    )
    source = confined(root, "run-state/discovery-tools/round028-source/" + revision)
    helper = confined(reader, "specs/015-learned-program-search/validation/candidate_source.py")
    runpy.run_path(str(helper))["verify_tree"](root, source, revision)
    return source


def original_grant(root, revision, grant_ref, minutes):
    if not re.fullmatch("sha256:[0-9a-f]{64}", grant_ref):
        raise ValueError("original qualification grant identity required")
    directory = confined(root, "run-state/discovery-authorizations")
    authority = confined(directory, "grant-" + grant_ref[7:] + ".json")
    grant = load(authority)
    folder = confined(root, "run-state/discovery/" + SESSION)
    if (
        grant["identity"] != grant_ref
        or grant["total_us"] != int(minutes) * 60_000_000
        or grant != load(confined(folder, "qualification/" + grant_ref[7:] + "/grant.json"))
    ):
        raise ValueError("original initialization grant, amount and journal owner required")
    validate_grant(
        grant,
        load(confined(directory, grant["authorizing_record_ref"][7:] + ".json")),
        root=physical_root_id(root),
        action="qualify",
        parent=prescore.parent(revision, load(confined(folder, checkpoints.MARKER))),
    )
    return authority


def diagnosis(detail):
    # Only emit finite public categories. Native stderr may contain paths, arguments or case data.
    for text, category in (
        ("No space left on device", "storage_full"),
        ("Disk quota exceeded", "storage_quota"),
        ("Permission denied", "permission_denied"),
        ("settled compatible runtime required", "runtime_unavailable"),
        ("local image store capacity unavailable", "store_unavailable"),
        ("above the physical reserve", "capacity_shortfall"),
        ("after holdback cannot fund", "insufficient_time"),
        ("external deadline", "deadline"),
        ("acquired initialization image no longer available", "runtime_unavailable"),
        ("qualified base package or candidate source archive changed", "base_or_archive_changed"),
        ("native runtime was built from different candidate inputs", "native_source_changed"),
        ("installed application differs", "installed_source_changed"),
        ("wheel manifest differs", "wheel_source_changed"),
        ("candidate wheel inventory differs", "wheel_source_changed"),
        ("No such file or directory", "missing_file"),
        ("bounded regular JSON required", "json_size_or_type"),
        ("original launch quote no longer reconciles", "quote_changed"),
        ("role network/root/mount isolation audit failed", "role_isolation"),
        ("ORIGINAL_IMAGE_OR_ROLE_MOUNTS_DIFFER", "role_isolation"),
        ("out of memory", "out_of_memory"),
    ):
        if text.lower() in detail.lower():
            return category
    return "unclassified"


def first_error(parent, *, private_log=False):
    path = confined(parent, "first-error.json", exists=False)
    if not path.is_file():
        return None
    error = load(path)
    if error != seal(error):
        raise ValueError("original first error identity differs")
    stage = error["stage"]
    kind = error["error"]
    if private_log:
        print(
            f"Original {stage} error ({kind}): {error['reason'][:2048]}",
            file=sys.stderr,
            flush=True,
        )
    return public_error(error)


def public_error(error):
    """Identify the exact private exception without exporting its message or arguments."""
    stage, kind = error["stage"], error["error"]
    return dict(
        identity=seal(error)["identity"],
        stage=stage if stage in (*PHASES, "qualification", "publication") else "unknown",
        error=(
            kind
            if kind
            in {
                "ValueError",
                "RuntimeError",
                "FileNotFoundError",
                "OSError",
                "PermissionError",
                "CalledProcessError",
                "InsufficientAuthority",
                "CapacityShortfall",
                "ReceiptGap",
                "VerificationExpired",
                "KeyError",
                "TypeError",
                "TimeoutExpired",
                "KeyboardInterrupt",
            }
            else "other"
        ),
        diagnosis=diagnosis(error["reason"]),
    )


def history(area, grant, *, private_log=False):
    """Take a single consistent read while the funding owner is idle. Never repair a journal here."""
    with confined(area.parent.parent, "run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        balance = BudgetJournal.inspect_existing(
            confined(area, "accounting"),
            grant["identity"],
            grant["resources"],
            include_windows=True,
        )
        operations = {}
        seen = set()
        for phase in PHASES:
            coords = checkpoints.coordinates(grant, phase)
            if phase in {"freeze_trials", "verify_paid_package", "publish_prescore"}:
                coords["method"] = "ordinary_initialization"
            logical = operation_key(coords, 0)["logical"]
            block = confined(area, "operations/" + logical[7:], exists=False)
            attempts = []
            receipt = None
            error = None
            build = {}
            if block.is_dir():
                error = first_error(block, private_log=private_log)
                pointer = confined(block, "receipt.json", exists=False)
                if pointer.is_file():
                    row = load(pointer)
                    key = f"{logical}/dispatch/{row['attempt']}"
                    if (
                        row != seal(row)
                        or row["coordinates"] != coords
                        or row["grant"] != grant["identity"]
                        or key not in balance["committed"]
                    ):
                        raise ValueError("operation receipt has no original settled dispatch")
                    receipt = row["identity"]
                    if phase == "build_runtime":
                        evidence = row["evidence"]
                        build["image_build"] = {
                            k: evidence[k] for k in ("image", "samples", "lowest")
                        }
            for ordinal in range(2):
                key = f"{logical}/dispatch/{ordinal}"
                if key not in balance["windows"]:
                    continue
                window = balance["windows"][key]
                out = confined(block, f"attempt-{ordinal:04d}", exists=False)
                child_name = {
                    "package": "package",
                    "package_runtime": "package",
                    "build_runtime": "image-build",
                }.get(phase)
                child = out / child_name if child_name else out
                code, category, log_hash = None, "unclassified", None
                if child.exists():
                    child = confined(area, child)
                    proc = confined(child, "process.json", exists=False)
                    if proc.is_file():
                        code = load(proc).get("returncode")
                        if code is not None and type(code) is not int:
                            raise ValueError("invalid supervisor exit code")
                    tail = log_tail(child, "stderr", limit=8192)
                    if tail:
                        category, log_hash = diagnosis(tail), digest(tail)
                        if private_log and (code not in (0, None) or error is not None):
                            print(
                                f"{phase} attempt {ordinal}, stderr tail:\n{tail}",
                                file=sys.stderr,
                                flush=True,
                            )
                if (
                    category == "unclassified"
                    and error
                    and load(confined(block, "first-error.json"))["attempt"] == ordinal
                ):
                    category = error["diagnosis"]
                seen.add(key)
                attempts.append(
                    dict(
                        ordinal=ordinal,
                        event=window["event"],
                        charged_us=window["charged_us"],
                        process_exit=code,
                        diagnosis=category,
                        stderr_digest=log_hash,
                    )
                )
            operations[phase] = dict(attempts=attempts, receipt=receipt, first_error=error, **build)
        if seen != balance["windows"].keys() or balance["consumed"]["tokens"]:
            raise ValueError("unexpected work in initialization grant")
        return dict(
            grant=grant["identity"],
            elapsed_us=balance["consumed"]["elapsed_us"],
            remaining_us=balance["remaining"]["elapsed_us"],
            first_error=first_error(area, private_log=private_log),
            operations=operations,
        )
