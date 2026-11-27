"""Funded donor continuation rehearsal. All training data and restored tensors stay private."""

from pathlib import Path
import time

from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.velocity_records import registered_worker
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_handoff import stage
from .progressive_roles import ROUND029_ROLE_POLICY
from .progressive_round028_receipts import verify_role_mounts
from .progressive_round028_rehearsal import physical_scope
from .progressive_velocity_continuation import verify_comparison
from .progressive_velocity_donor import donated_ordinary, PILOT_REPORT
from .progressive_velocity_operations import InsufficientAuthority, run_operation


def donation_parent(revision):
    return digest([velocity_policy(), revision, PILOT_REPORT, "ordinary_4M_disposable_next_update"])


def authorize_donation_check(root, minutes, revision):
    from .progressive_velocity_qualification import authorize_qualification

    data, _ = donated_ordinary(root)
    return authorize_qualification(root, minutes, donation_parent(revision),
                                 {"tokens": 2 * data["plan"]["update_tokens"], "attempts": 2})


def coordinates(grant, step, seed=0):
    return dict(study=grant["identity"], stage="qualification", method="00", seed=seed, look=4_000_000,
                panel="training", profile="ordinary_next_update", input_mode="true", role=step, ordinal=0)


def check_request(request, binding, data):
    if (request != seal(request) or request["version"] != "round029-donation-request/v1"
            or request["binding"] != binding["identity"] or request["source"] != binding["source"]
            or request["role_policy"] != ROUND029_ROLE_POLICY or any(request[k] != v for k,v in data.items())):
        raise ValueError("original donation, worker, or training projection differs")
    worker = dict(source=binding["candidate"], image=request["image"], device="hip", role_policy=ROUND029_ROLE_POLICY)
    registered_worker(worker, worker)
    return request


def check_stage(area, out, request, mounts):
    produced = load(confined(out, "producer.json"))
    if (produced != seal(produced) or produced["version"] != "round029-donation-result/v1"
            or produced["request"] != request["identity"] or produced["device"] != "hip"):
        raise ValueError("original installed donation response required")
    verify_comparison(produced["comparison"], request["plan"])
    expected = {dest: (path, False) for path,dest in mounts}
    expected.update({"/work/inputs/request.json": (area / "request.json", False), "/work/output": (out, True)})
    verify_role_mounts(load(confined(out, "container-inspect.json")), request["image"], expected)
    process, cleanup, lease = [load(confined(out, name)) for name in ("process.json", "cleanup.json", "gpu-lease.json")]
    if (process["outcome"] != "COMPLETE" or process["returncode"] != 0 or process["cleanup_failure"]
            or cleanup["acknowledged"] is not True or cleanup["within_bound"] is not True
            or type(lease["elapsed_us"]) is not int or type(produced["elapsed_us"]) is not int
            or not 0 < produced["elapsed_us"] <= lease["elapsed_us"]
            or type(produced["peak_device_bytes"]) is not int or produced["peak_device_bytes"] <= 0):
        raise ValueError("original worker termination, HIP lease or measured memory differs")
    return seal(dict(request=request["identity"], producer=produced["identity"], comparison=produced["comparison"],
                     lease_us=lease["elapsed_us"], peak_device_bytes=produced["peak_device_bytes"],
                     artifacts=[reference(area, out / name) for name in
                                ("producer.json", "container-inspect.json", "process.json", "cleanup.json", "gpu-lease.json")]))


def qualify(args, root, folder, source, revision, grant, host_factory):
    from .progressive_round029 import result

    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    binding = seal(dict(version="round029-donation/v1", grant=grant["identity"], candidate=revision,
                        source=digest(collect_snapshot(source)), pilot=PILOT_REPORT))
    retain(area / "binding.json", binding)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    base = dict(area=area, journal=journal, grant=grant, minimum_us=121_000_000)
    host = None
    donation = None

    def get_host():
        nonlocal host
        if host is None:
            host = host_factory(args, source_root=source, workspace=root, revision=revision)
        return host

    def original():
        nonlocal donation
        if donation is None:
            donation = donated_ordinary(root)
        return donation

    def seconds_left(deadline):
        seconds = (deadline - time.monotonic_ns()) // 1_000_000_000 - 120
        if seconds < 1:
            raise InsufficientAuthority("remaining allowance cannot fund launch and the 120-second cleanup window")
        return min(seconds, 86400)

    def prepare(out, reserve):
        deadline = time.monotonic_ns() + reserve * 1000
        data, _ = original()
        worker = get_host()
        if (area / "request.json").exists():
            pinned = load(confined(area, "request.json"))
            image = worker.engine.image(pinned["image"])
        elif args.image:
            image = worker.engine.image(args.image)
        else:
            image = worker.engine.build(source, target="gpu", tag="oeis-learn:round029-" + revision,
                                        base_image=args.gpu_base_image, seconds=seconds_left(deadline))
        physical = physical_scope(worker, "hip", machine=True)
        request = seal(dict(version="round029-donation-request/v1", binding=binding["identity"],
                            source=binding["source"], image=image, physical=physical, role_policy=ROUND029_ROLE_POLICY, **data))
        check_request(request, binding, data)
        retain(area / "request.json", request)
        retain(out / "producer.json", seal(dict(request=request["identity"])))
        # All preparation subprocesses have synchronously terminated at this host boundary.
        retain(out / "cleanup.json", dict(acknowledged=True, within_bound=time.monotonic_ns() <= deadline))

    def prepared(out):
        request = check_request(load(confined(area, "request.json")), binding, original()[0])
        producer = load(confined(out, "producer.json"))
        if producer != seal(dict(request=request["identity"])):
            raise ValueError("original preparation receipt differs")
        return producer

    def dispatch(out, reserve):
        deadline = time.monotonic_ns() + reserve * 1000
        worker = get_host()
        if worker.engine.image(request["image"]) != request["image"] or physical_scope(worker, "hip", machine=True) != request["physical"]:
            raise ValueError("registered physical donor or immutable worker image changed")
        stage(worker, source, request["image"], out, ["python", "-I", "-m",
              "oeis_learn.pipeline_validation.progressive_velocity_continuation"], gpu=True, record_lease=True,
              seconds=seconds_left(deadline), readonly_inputs=[(area / "request.json", "/work/inputs/request.json"), *mounts])

    def status(state, reason, next_action, **extra):
        snapshot = journal.snapshot()
        if (area / "first-error.json").exists():
            extra["first_error"] = load(confined(area, "first-error.json"))
        value = result("qualify", state, reason, next_action, remaining=snapshot["remaining"]["elapsed_us"],
                       stage="qualification", scope="donated_ordinary_next_update",
                       charged_tokens=snapshot["consumed"]["tokens"], remaining_tokens=snapshot["remaining"]["tokens"],
                       elapsed_us=snapshot["consumed"]["elapsed_us"], research_dispatches=0, **extra)
        path = folder / "status.json"
        if not path.exists() or load(confined(folder, "status.json")) != seal(value):
            revise(path, seal(value))
        return value

    step = "prepare"
    try:
        setup = run_operation(**base, coordinates=coordinates(grant, step), request=binding, cost={},
                              execute=prepare, verify=prepared)
        request = load(confined(area, "request.json"))
        _, mounts = original()
        step = "train"
        cost = dict(tokens=2 * request["plan"]["update_tokens"], attempts=2)
        receipt = run_operation(**base, coordinates=coordinates(grant, step, request["parent"]["milestone"]["seeds"]["initialization"]),
                                request=request, cost=cost,
                                execute=dispatch, verify=lambda out: check_stage(area, out, request, mounts))
        record = seal(dict(binding=binding["identity"], preparation=setup["identity"], operation=receipt["identity"],
                           attempt=receipt["attempt"], **{k:receipt["evidence"][k] for k in
                                                        ("comparison", "lease_us", "peak_device_bytes")}))
        path = area / "receipt.json"
        retain(path, record)
        return status("COMPLETE", "Verified one ordinary next update through both installed policies",
                      "verify", last=record["identity"], receipt=record, receipt_path=str(path.relative_to(root))), 0
    except (Exception, KeyboardInterrupt) as exc:
        err = dict(stage=step, error=type(exc).__name__, reason=str(exc) or type(exc).__name__)
        if not (area / "first-error.json").exists():
            retain(area / "first-error.json", seal(err))
        blocked = isinstance(exc, InsufficientAuthority)
        return status("BLOCKED" if blocked else "FAILED", err["reason"], "inspect_attempt"), (
            2 if blocked else 130 if isinstance(exc, KeyboardInterrupt) else 1)
