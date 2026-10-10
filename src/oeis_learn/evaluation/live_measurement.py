"""Single engineering session and diagnostic/paired pilot through isolated role views."""

from pathlib import Path
import time
from oeis_learn.experiments.artifacts import (
    canonical_bytes,
    atomic_write,
    ArtifactPath,
    compute_canonical_digest as digest,
)
from oeis_learn.experiments.decisions import PilotSchedule, useful_effect, decision_records
from oeis_learn.experiments.pilot_models import pilot_profile
from oeis_learn.tracking.foundation_metrics import EventLog
from oeis_learn.tracking.role_coordinator import RoleCoordinator, copy_view


def write_view(root, values, files=None):
    root = Path(root)
    copy_view(root, files or {})
    for name, value in values.items():
        atomic_write(ArtifactPath(root, name), canonical_bytes(value))
    return root


def execute_device(
    manifest,
    assets,
    source_root,
    output,
    *,
    coordinator_factory=RoleCoordinator,
    clock=time.monotonic_ns,
):
    """Budgets include setup/checkpoint/evaluation; work never receives final truth."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    log = EventLog(root / "live-events")
    if log.events:
        raise RuntimeError("previous live issuance cannot be repeated/refunded")
    log.append("issued", dict(manifest_sha256=digest(manifest), started_ns=time.time_ns()))
    started = clock()
    coordinator = coordinator_factory(root / "coordination", manifest["assets"]["image"], workers=8)
    common = {"examples.json": assets["examples"]}
    settings = dict(seed=7001, device="cuda", profile=pilot_profile())
    try:
        # Real-device prerequisites are checked in the GPU role, with gate files
        # but no evaluator data. Host-side status strings cannot authorize work.
        gate_files = {}
        for ref in assets["gate_evidence"]:
            name = ref["path"][len("assets/") :]
            gate_files[name] = Path(source_root) / ref["path"]
        inputs = write_view(
            root / "probe-input",
            dict(
                common,
                **{
                    "settings.json": settings,
                    "gate7.json": assets["gate7"],
                    "gate8.json": assets["gate8"],
                    "baseline.json": assets["baseline"],
                    "baseline-checkpoint.json": assets["baseline_checkpoint"],
                },
            ),
            gate_files,
        )
        probe = coordinator.run(
            "probe", role="learner", inputs=inputs, output=root / "probe", seconds=600, gpu=True
        )
        if (
            probe["status"] != "PASS"
            or not probe["hip"]
            or not probe["actual_parameter_device"].startswith("cuda")
            or probe["code_sha256"] != manifest["identities"]["code"]
        ):
            raise ValueError("actual device correctness gate failed")
        log.append("probe", probe)
        # The two-hour clock started before role setup and device gate.
        remaining = (7200 * 10**9 - (clock() - started)) / 1e9
        if remaining <= 0:
            raise TimeoutError("engineering cap exhausted")
        inputs = write_view(root / "engineering-input", dict(common, **{"settings.json": settings}))
        engineering = coordinator.run(
            "engineering",
            role="learner",
            inputs=inputs,
            output=root / "engineering",
            seconds=remaining,
            gpu=True,
        )
        log.append(
            "engineering", dict(result_sha256=digest(engineering), charged_ns=clock() - started)
        )
        pilot = run_pilot(manifest, assets, root, coordinator, clock=clock)
        return dict(
            engineering=engineering,
            pilot=pilot,
            device=probe,
            status="complete",
            executed=True,
            qualified=True,
            charged_ns=clock() - started,
        )
    finally:
        coordinator.close()


def run_pilot(manifest, assets, root, coordinator, *, clock=time.monotonic_ns, utc_clock=None):
    """One selection, matched fresh seed arms and a common timed evaluation schedule."""
    root = Path(root)
    schedule = PilotSchedule(root / "pilot-events", digest(manifest), clock=utc_clock)
    try:
        return _run_pilot(manifest, assets, root, coordinator, schedule, clock)
    except BaseException as exc:
        schedule.interrupt(f"{type(exc).__name__}: {exc}")
        raise


def _run_pilot(manifest, assets, root, coordinator, schedule, clock):
    task = schedule.issue()
    settings = dict(
        seed=task["seed"],
        device="cuda",
        profile=pilot_profile(),
        seconds=task["seconds"] - 300,
        update_cap=1000,
    )
    inputs = write_view(
        root / "diagnostic-input", {"settings.json": settings, "examples.json": assets["examples"]}
    )
    result = coordinator.run(
        "diagnostic",
        role="learner",
        inputs=inputs,
        output=root / "diagnostic",
        seconds=task["seconds"],
        gpu=True,
    )
    schedule.complete()
    diagnostic = result["diagnostic"]
    selected, _ = schedule.select(diagnostic)
    arms = []
    if selected is not None:
        for seed in (7001, 7002, 7003):
            if seed != 7001 and not useful_effect(*schedule.pairs[0], *schedule.durations[0]):
                break
            for arm in ("baseline", "challenger"):
                task = schedule.issue()
                if task["seed"] != seed or task["arm"] != arm:
                    raise ValueError("paired issuance order differs")
                arm_root = root / f"{selected}-{seed}-{arm}"
                arm_root.mkdir()
                profile = pilot_profile(
                    selected, arm, binding_scope_cause=diagnostic.get("binding_scope_cause", False)
                )
                arm_start, previous, evaluations = clock(), None, []
                # Each boundary reserves work *inside* the55-minute arm for
                # checkpoint, generation and sealed scoring. No best-from-future selection.
                for minute in (15, 30, 45, 55):
                    deadline = arm_start + minute * 60 * 10**9
                    remaining = (deadline - clock()) / 1e9
                    if remaining <= 0:
                        break
                    settings = dict(
                        seed=seed,
                        device="cuda",
                        profile=profile,
                        seconds=max(5, remaining - 180),
                        update_cap=1000000,
                    )
                    stage = arm_root / f"minute-{minute}"
                    stage.mkdir()
                    inputs = write_view(
                        stage / "train-input",
                        {"settings.json": settings, "examples.json": assets["examples"]},
                        {
                            "state.pt": previous,
                            "state-manifest.json": previous.parent / "state-manifest.json",
                        }
                        if previous
                        else {},
                    )
                    trained = coordinator.run(
                        "train",
                        role="learner",
                        inputs=inputs,
                        output=stage / "train",
                        seconds=max(0.001, remaining - 120),
                        gpu=True,
                    )
                    previous = stage / "train/state.pt"
                    visible = dict(
                        groups=[
                            dict(group_id=g["group_id"], values=g["values"][:20])
                            for g in assets["development"]["groups"]
                        ]
                    )
                    inputs = write_view(
                        stage / "generate-input",
                        {"settings.json": settings, "visible.json": visible},
                        {
                            "state.pt": previous,
                            "state-manifest.json": previous.parent / "state-manifest.json",
                        },
                    )
                    remaining = (deadline - clock()) / 1e9
                    if remaining <= 0:
                        break
                    seal = coordinator.run(
                        "generate",
                        role="generation",
                        inputs=inputs,
                        output=stage / "generate",
                        seconds=remaining,
                        gpu=True,
                    )
                    # The learner/generator is gone before private scoring is mounted.
                    inputs = write_view(
                        stage / "evaluate-input",
                        {"seal.json": seal, "development.json": assets["development"]},
                    )
                    remaining = (deadline - clock()) / 1e9
                    if remaining <= 0:
                        break
                    score = coordinator.run(
                        "evaluate",
                        role="evaluator",
                        inputs=inputs,
                        output=stage / "evaluate",
                        seconds=remaining,
                        gpu=False,
                    )
                    if score["selection_split"] != "development" or score["final_truth_available"]:
                        raise ValueError("evaluation boundary failed")
                    evaluations.append(
                        dict(
                            minute=minute,
                            score=score,
                            checkpoint_sha256=trained["state_sha256"],
                            elapsed_ns=clock() - arm_start,
                            unique_groups_per_charged_hour=len(score["solved_groups"])
                            * 3600e9
                            / max(1, clock() - arm_start),
                        )
                    )
                    atomic_write(
                        ArtifactPath(stage, "evaluation.json"), canonical_bytes(evaluations[-1])
                    )
                if [e["minute"] for e in evaluations] != [15, 30, 45, 55]:
                    raise TimeoutError("incomplete fixed evaluation schedule; arm stays charged")
                # Primary comparison uses the last completed *common* boundary;
                # best checkpoint selection is development-only and separate.
                arms.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        profile_sha256=profile["identity_sha256"],
                        evaluations=evaluations,
                        best_minute=max(
                            evaluations,
                            key=lambda e: (len(e["score"]["solved_groups"]), -e["minute"]),
                        )["minute"],
                    )
                )
                if arm == "challenger":
                    baseline = arms[-2]
                    common = sorted(
                        set(e["minute"] for e in baseline["evaluations"])
                        & set(e["minute"] for e in evaluations)
                    )
                    if not common:
                        raise ValueError("no matched evaluation compute")
                    minute = common[-1]
                    b = next(e for e in baseline["evaluations"] if e["minute"] == minute)
                    c = next(e for e in evaluations if e["minute"] == minute)
                    if minute != 55 or len(b["score"]["solved_groups"]) != schedule.pairs[-1][0]:
                        raise ValueError(
                            "last completed evaluation boundaries differ; comparison inconclusive"
                        )
                    count = len(c["score"]["solved_groups"])
                else:
                    count = len(evaluations[-1]["score"]["solved_groups"])
                schedule.complete(solved_groups=count)
    pairs = [p for p in schedule.pairs if p[1] is not None]
    profiles = {
        q: (
            pilot_profile(
                q, "baseline", binding_scope_cause=diagnostic.get("binding_scope_cause", False)
            )["identity_sha256"],
            pilot_profile(
                q, "challenger", binding_scope_cause=diagnostic.get("binding_scope_cause", False)
            )["identity_sha256"],
        )
        for q in manifest["protocol"]["priority"]
    }
    records = decision_records(
        manifest["identities"],
        diagnostic,
        pairs,
        charged_ns=schedule.charged,
        durations=schedule.durations,
        profile_identities=profiles,
    )
    report = dict(
        version="conditional-pilot/v1",
        selected=selected,
        diagnostic=diagnostic,
        diagnostic_sha256=digest(result),
        arms=arms,
        pairs=pairs,
        paired_durations_ns=schedule.durations,
        decisions=records,
        charged_ns=schedule.charged,
        seconds_cap=21600,
        final_truth_available=False,
    )
    atomic_write(ArtifactPath(root, "pilot.json"), canonical_bytes(report))
    return report


def validate_live_evidence(value, manifest, root):
    """Replay device, fixed protocol and role-owned results before accepting a label."""
    from oeis_learn.experiments.artifacts import load_json, compute_file_hash

    root = Path(root)
    if (
        value.get("status") != "complete"
        or value.get("executed") is not True
        or value.get("qualified") is not True
    ):
        raise ValueError("complete executed live measurement required")
    for key, directory in (("device", "probe"), ("engineering", "engineering")):
        recorded = load_json((root / directory / "result.json").read_bytes())
        if recorded != value[key]:
            raise ValueError("role result differs from measurement evidence")
    device = value["device"]
    if device.get("code_sha256") != manifest["identities"]["code"]:
        raise ValueError("executed image code differs from frozen manifest")
    if (
        not device.get("hip")
        or not device.get("actual_parameter_device", "").startswith("cuda")
        or len(device.get("updates", [])) != 3
    ):
        raise ValueError("actual HIP measured updates required")
    engineering = value["engineering"]
    if (
        engineering.get("status") != "complete"
        or engineering.get("prompts") != 128
        or engineering.get("seeds") != [7001, 7002, 7003]
    ):
        raise ValueError("fixed engineering denominator/seed evidence missing")
    results = engineering["results"]
    if set(results) != {
        "inference",
        "grammar",
        "workers",
        "fuel_streaming",
        "relations",
        "precision",
    }:
        raise ValueError("complete engineering comparisons required")
    for seed in (7001, 7002, 7003):
        variants = results["inference"][str(seed)]
        if set(variants) != {"cache", "batch", "attention"}:
            raise ValueError("comparison missing")
        for record in variants.values():
            if (
                record["warmups"] != 10
                or record["repeats"] != 30
                or not record["semantic_equivalence"]
                or any(len(t) != 30 for t in record["timings"].values())
            ):
                raise ValueError("paired fixed engineering repeats/equivalence required")
    pilot = value["pilot"]
    if (
        load_json((root / "pilot.json").read_bytes()) != pilot
        or pilot["final_truth_available"]
        or pilot["seconds_cap"] != 21600
    ):
        raise ValueError("pilot result/cap differs")
    schedule = PilotSchedule(root / "pilot-events", digest(manifest))
    if (
        schedule.interrupted
        or schedule.pending
        or schedule.charged != pilot["charged_ns"]
        or schedule.pairs != pilot["pairs"]
        or schedule.durations != pilot["paired_durations_ns"]
    ):
        raise ValueError("pilot durable issuance/completion differs")
    for arm in pilot["arms"]:
        if [e["minute"] for e in arm["evaluations"]] != [15, 30, 45, 55]:
            raise ValueError("fixed development schedule missing")
        directory = root / f"{pilot['selected']}-{arm['seed']}-{arm['arm']}"
        for evaluation in arm["evaluations"]:
            stage = directory / f"minute-{evaluation['minute']}"
            if compute_file_hash(stage / "train/state.pt") != evaluation["checkpoint_sha256"]:
                raise ValueError("evaluated checkpoint changed")
            if load_json((stage / "evaluation.json").read_bytes()) != evaluation:
                raise ValueError("scheduled immutable evaluation changed")
            score = load_json((stage / "evaluate/result.json").read_bytes())
            if (
                score != evaluation["score"]
                or score["final_truth_available"]
                or score["selection_split"] != "development"
            ):
                raise ValueError("development evidence changed")
    events = EventLog(root / "coordination/roles").events
    starts = [key for key, r in events.items() if "operation" in r]
    if not starts or any(
        events.get(key + "-stop", {}).get("reclaimed") is not True for key in starts
    ):
        raise ValueError("role reclamation evidence missing")
    if value["charged_ns"] > (7200 + 21600) * 10**9:
        raise ValueError("combined measurement cap exceeded")
    return True
