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


class VerificationExpired(InsufficientAuthority):
    pass


def funded_verification(verify, out, deadline_us):
    """A pure host reader can block on I/O. Give it a separately terminable process group."""
    import json
    import multiprocessing
    import os
    import signal
    import tempfile

    def remaining():
        return (deadline_us - time.monotonic_ns() // 1000) / 1_000_000

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as response:
        def read():
            os.setsid()
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, remaining()))
            try:
                value = dict(evidence=verify(out))
            except Exception as exc:
                value = dict(error=type(exc).__name__, reason=str(exc))
            json.dump(value, response, allow_nan=False)
            response.flush()

        if remaining() <= 0:
            raise VerificationExpired("funded verification reached its deadline")
        reader = multiprocessing.get_context("fork").Process(target=read)
        reader.start()
        try:
            available = max(0, remaining())
            reader.join(max(0, available - min(0.05, available / 10)))
            if reader.is_alive() or reader.exitcode == -signal.SIGALRM:
                raise VerificationExpired("funded verification reached its deadline")
            if reader.exitcode:
                raise ValueError(f"verification process exited with status {reader.exitcode}")
            response.seek(0)
            result = json.load(response)
            if "error" in result:
                raise ValueError(result["error"] + ": " + result["reason"])
            return result["evidence"]
        finally:
            if reader.is_alive():
                try:
                    os.killpg(reader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    reader.kill()
            reader.join()
            reader.close()


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


def check_reserved_work(snapshot, prefix, work):
    for key, debit in snapshot.get("reserved_work", {}).items():
        if key.startswith(prefix) and any(debit[k] != v for k, v in work.items() if k != "elapsed_us"):
            raise ValueError("original dispatch debit differs from the registered work")


def replay_operation(area, coordinates, request, grant, cost, snapshot, verify):
    """Read and rederive a completed original operation, without opening a writable journal."""
    logical = operation_key(coordinates, 0)["logical"]
    block = confined(area, "operations/" + logical[7:])
    work = vector(dict(dict.fromkeys(KEYS, 0), **cost))
    if load(confined(block, "input.json")) != input_record(coordinates, request, work, grant):
        raise ValueError("original registered operation input differs")
    check_reserved_work(snapshot, logical + "/dispatch/", work)
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
    installed=None,
    before_dispatch=None,
    protected_us=0,
    maximum_us=None,
):
    """Only a verified supervisor boundary may release unused time. Issued work stays charged."""
    if (
        request != seal(request)
        or coordinates["study"] != grant["identity"]
        or journal.binding != {"campaign": grant["identity"], "caps": grant["resources"]}
        or "elapsed_us" in cost
        or type(protected_us) is not int
        or protected_us < 0
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
    state = journal.snapshot(include_work=True)
    check_reserved_work(state, prefix, work)
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
            available = installed(out) if installed else (out / "producer.json").is_file()
            if available and (out / "cleanup.json").is_file():
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
    reserve = left["elapsed_us"] - grant["holdback_us"] - protected_us
    if maximum_us is not None:
        if type(maximum_us) is not int or maximum_us < 1:
            raise ValueError("positive supervisor window required")
        reserve = min(maximum_us, reserve)
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
    if before_dispatch is not None:
        before_dispatch()
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
        evidence = funded_verification(verify, out, start + reserve)
        proof = operation_receipt(coordinates, request, grant, attempt, evidence)
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
        expired = isinstance(exc, VerificationExpired)
        if expired and acknowledged:
            retain(block / f"recovering-{attempt:04d}.json", seal(dict(reservation=reservation["key"])))
        end = time.monotonic_ns() // 1000
        journal.commit(
            reservation["key"],
            boot=boot,
            monotonic_us=end,
            termination_proven=acknowledged and not expired and end - start <= reserve,
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
