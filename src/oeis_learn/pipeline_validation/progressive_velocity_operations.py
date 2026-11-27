"""Debit and reopen one registered operation under the physical session owner's lock."""

from pathlib import Path
import time

from oeis_learn.generalization.progressive.budgets import KEYS, vector
from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.generalization.progressive.velocity_records import operation_key
from .identity import confined
from .preparation_recovery import retain


class InsufficientAuthority(ValueError):
    pass


def input_record(coordinates, request, work, grant):
    return seal(
        dict(
            coordinates=coordinates,
            request=request,
            work=work,
            grant=grant["identity"],
            holdback_us=grant["holdback_us"],
        )
    )


def operation_receipt(coordinates, request, grant, attempt, item):
    if item != seal(item):
        raise ValueError("unsealed operation evidence")
    return seal(
        dict(
            coordinates=coordinates,
            request=request["identity"],
            grant=grant["identity"],
            operation=operation_key(coordinates, attempt),
            attempt=attempt,
            evidence=item,
        )
    )


def replay_operation(area, coordinates, request, grant, cost, snapshot, verify):
    """Read and rederive a completed original operation, without opening a writable journal."""
    logical = operation_key(coordinates, 0)["logical"]
    block = confined(area, "operations/" + logical[7:])
    work = vector(dict(dict.fromkeys(KEYS, 0), **cost))
    if load(confined(block, "input.json")) != input_record(coordinates, request, work, grant):
        raise ValueError("original registered operation input differs")
    prior = load(confined(block, "receipt.json"))
    attempt = prior["attempt"]
    if f"{logical}/dispatch/{attempt}" not in snapshot["committed"]:
        raise ValueError("original dispatch debit missing")
    out = confined(block, f"attempt-{attempt:04d}")
    proof = operation_receipt(coordinates, request, grant, attempt, verify(out))
    boundary = load(confined(block, f"boundary-{attempt:04d}.json"))
    if prior != proof or boundary != seal(boundary) or boundary["result"] != proof["identity"]:
        raise ValueError("persisted operation differs from original dispatch")
    return proof


def run_operation(
    *,
    area,
    journal,
    grant,
    coordinates,
    request,
    cost,
    execute,
    verify,
    minimum_us=1,
    max_attempts=None,
):
    """Only a verified supervisor boundary may release unused time. Issued work stays charged."""
    if (
        request != seal(request)
        or coordinates["study"] != grant["identity"]
        or journal.binding != {"campaign": grant["identity"], "caps": grant["resources"]}
        or "elapsed_us" in cost
        or not set(cost) <= set(KEYS)
    ):
        raise ValueError("operation requires its original grant, request and registered work")
    work = vector(dict(dict.fromkeys(KEYS, 0), **cost))
    logical = operation_key(coordinates, 0)["logical"]
    block = confined(area, "operations/" + logical[7:], exists=False)
    block.mkdir(parents=True, exist_ok=True)
    retain(block / "input.json", input_record(coordinates, request, work, grant))
    prefix = logical + "/dispatch/"

    def verified(out, attempt):
        return operation_receipt(coordinates, request, grant, attempt, verify(out))

    receipt = block / "receipt.json"
    state = journal.reconcile()
    keys = [k for k in state["pending"] + state["committed"] if k.startswith(prefix)]
    # A crash may separate the checked result, the irreversible debit and the published pointer.
    # The durable host boundary carries the original monotonic end for that same reservation.
    for attempt in range(len(keys)):
        key = prefix + str(attempt)
        if key not in keys:
            raise ValueError("gap in physical operation attempts")
        out = confined(block, f"attempt-{attempt:04d}", exists=False)
        # The worker can write only its attempt directory. Supervisor proofs live outside that mount.
        boundary = confined(block, f"boundary-{attempt:04d}.json", exists=False)
        recovery = confined(block, f"recovering-{attempt:04d}.json", exists=False)
        if boundary.exists():
            recorded = load(boundary)
            proof = verified(out, attempt)
            if recorded != seal(recorded) or recorded["result"] != proof["identity"]:
                raise ValueError("installed operation or original supervisor boundary differs")
            journal.commit(
                key,
                boot=recorded["boot"],
                monotonic_us=recorded["end_us"],
                termination_proven=recorded["within_reservation"],
            )
            retain(receipt, proof)
            return proof
        if key in state["pending"] or recovery.exists():
            # A dead supervisor cannot prove unused time. Still salvage an installed result
            # with independently verified process termination and a complete original debit.
            # Record the recovery intent before committing so a second interruption is resumable.
            if out.is_dir():
                retain(recovery, seal(dict(reservation=key)))
            journal.commit(key)
            if (out / "producer.json").is_file() and (out / "cleanup.json").is_file():
                cleanup = load(confined(out, "cleanup.json"))
                if cleanup.get("acknowledged") is True and cleanup.get("within_bound") is True:
                    proof = verified(out, attempt)
                    retain(
                        boundary,
                        seal(
                            dict(
                                result=proof["identity"],
                                boot=None,
                                end_us=None,
                                within_reservation=False,
                            )
                        ),
                    )
                    retain(receipt, proof)
                    return proof
    if receipt.exists():
        raise ValueError("original operation receipt has lost its verified boundary")

    left = journal.snapshot()["remaining"]
    reserve = left["elapsed_us"] - grant["holdback_us"]
    if reserve < minimum_us:
        raise InsufficientAuthority(
            "remaining workstation time after holdback cannot fund the minimum launch window"
        )
    if any(left[k] < amount for k, amount in work.items()):
        raise InsufficientAuthority("original remaining resources cannot fund the operation")
    attempt = len(keys)
    if max_attempts is not None and attempt >= max_attempts:
        raise InsufficientAuthority(
            "the authorized dispatch attempts for this operation are exhausted"
        )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    start = time.monotonic_ns() // 1000
    reservation = journal.reserve(
        logical, "dispatch", attempt, dict(work, elapsed_us=reserve), boot=boot, monotonic_us=start
    )
    out = confined(block, f"attempt-{attempt:04d}", exists=False)
    acknowledged = False
    try:
        out.mkdir()
        execute(out, reserve)
        proof = verified(out, attempt)
        cleanup = load(confined(out, "cleanup.json"))
        if cleanup.get("acknowledged") is not True or cleanup.get("within_bound") is not True:
            raise ValueError("original supervisor termination and cleanup required")
        acknowledged = True
        end = time.monotonic_ns() // 1000
        within = 0 <= end - start <= reserve
        if not within:
            raise InsufficientAuthority("operation exceeded its original wall-time reservation")
        retain(
            block / f"boundary-{attempt:04d}.json",
            seal(dict(result=proof["identity"], boot=boot, end_us=end, within_reservation=within)),
        )
        journal.commit(reservation["key"], boot=boot, monotonic_us=end, termination_proven=True)
        retain(receipt, proof)
        return proof
    except (Exception, KeyboardInterrupt) as exc:
        # A failed but demonstrably stopped subprocess still pays its complete non-time debit.
        try:
            cleanup = load(confined(out, "cleanup.json"))
            acknowledged = (
                cleanup.get("acknowledged") is True and cleanup.get("within_bound") is True
            )
        except (OSError, ValueError, TypeError):
            pass
        end = time.monotonic_ns() // 1000
        journal.commit(
            reservation["key"],
            boot=boot,
            monotonic_us=end,
            termination_proven=acknowledged and end - start <= reserve,
        )
        path = block / "first-error.json"
        if not path.exists():
            retain(
                path,
                seal(
                    dict(
                        attempt=attempt,
                        stage=coordinates["stage"],
                        reason=str(exc) or type(exc).__name__,
                        error=type(exc).__name__,
                    )
                ),
            )
        raise
