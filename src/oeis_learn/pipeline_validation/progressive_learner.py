"""Installed finite learner; public supervisor clocks, isolated training bytes only."""

import copy
import time
from pathlib import Path

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import validate_campaign, validate_task
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    seal,
)
from oeis_learn.generalization.progressive.curriculum import Controller
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.pipeline_validation.progressive_training import (
    Heartbeat,
    Progress,
    RecoverableUpdate,
)


def compatibility(package):
    profile = package["profile"]
    model = {k: v for k, v in profile["model"].items() if k != "seed"}
    policy = {
        k: v for k, v in package["update_policy"].items() if k not in {"identity", "model_profile"}
    }
    value = {
        "model": model,
        "shape": profile["shape"],
        "inputs": profile["inputs"],
        "codec": profile["codec"],
        "encoder_positions": profile["encoder_positions"],
        "decoder_positions": profile["decoder_positions"],
        "policy": policy,
        "shards": package["shards"],
        "layout": package["layout"],
        "refresh_updates": package["refresh_updates"],
        "scope": package["scope"],
    }
    if "oeis_replay" in package or "oeis_rules" in package:
        from oeis_learn.generalization.progressive.replay import RULE

        value["replay_policy"] = RULE
    if "component_distillation" in package or "component_rules" in package:
        from oeis_learn.generalization.progressive.component_distillation import RULE

        value["component_policy"] = RULE
    return digest(value)


def validate_input(package, task, *, allow_pending=False):
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.progressive.models import validate_profile
    from oeis_learn.pipeline_validation.progressive_training import validate_update_policy

    if (
        set(package)
        - {
            "weights",
            "full_state",
            "origin",
            "oeis_replay",
            "oeis_rules",
            "component_distillation",
            "component_rules",
        }
        != {
            "version",
            "identity",
            "profile",
            "update_policy",
            "shards",
            "layout",
            "refresh_updates",
            "scope",
            "diagnostic_fault",
        }
        or package["version"] != "progressive-training-input/v1"
        or seal(package) != package
    ):
        raise ValueError("closed frozen training input required")
    if (
        task["purpose"] == "research"
        and task["payload"].get("stage") == "scaling"
        and (
            package["update_policy"]["schedule_horizon_tokens"] != 400000000
            or package["update_policy"]["warmup_tokens"] != (
                1250000 if package["update_policy"]["schedule"] == "warmup_cosine" else 0
            )
        )
    ):
        raise ValueError("scaling must retain the frozen 400M trajectory and schedule-specific warmup")
    if "oeis_rules" in package:
        from oeis_learn.generalization.progressive.replay import RULE

        if package["oeis_rules"] != RULE or ("oeis_replay" not in package and not allow_pending):
            raise ValueError("frozen OEIS mixture requires its committed archive cut")
    if "oeis_replay" in package:
        from oeis_learn.generalization.progressive.replay import Mixer

        mixer = Mixer(package["oeis_replay"])
        if (
            task["track"] != "oeis_derived"
            or task["payload"]["arm"] != mixer.package["arm"]
            or any(a["purpose"] != task["purpose"] for a in mixer.package["admissions"])
        ):
            raise ValueError("separate paired OEIS-derived replay authority required")
    if "component_rules" in package or "component_distillation" in package:
        from oeis_learn.generalization.progressive.component_distillation import (
            RULE,
            ComponentMixer,
        )

        if "oeis_rules" in package or "oeis_replay" in package:
            raise ValueError("separate component/OEIS training profiles required")
        if package.get("component_rules", RULE) != RULE or (
            "component_distillation" not in package and not allow_pending
        ):
            raise ValueError("frozen component policy requires its committed task cut")
        if "component_distillation" in package:
            selected = package["component_distillation"]
            ComponentMixer(selected)
            if (
                task["payload"]["stage"] != "components"
                or task["payload"]["arm"] != selected["arm"]
            ):
                raise ValueError("paired component arm/task authority differs")
            if selected["library"] is not None and (
                selected["library"]["library"]["track"] != task["track"]
                or selected["library"]["library"]["purpose"] != task["purpose"]
            ):
                raise ValueError("component library track/purpose differs")
    initializer = task["payload"]["initializer"]
    if (
        initializer.get("mode") == "weights_only"
        and package.get("weights") != initializer["artifact"]
    ):
        raise ValueError("weights-only initializer must bind the frozen artifact")
    if (
        initializer.get("mode") not in {None, "weights_only", "full_state"}
        and package.get("weights") is not None
    ):
        raise ValueError("unexpected imported weights on a fresh/full-state task")
    if initializer.get('mode')=='full_state' and initializer['artifact'] is not None:
        if package.get('full_state',{}).get('artifact')!=initializer['artifact']:
            raise ValueError('direct full-state initializer must bind original source closure')
    elif 'full_state' in package:
        raise ValueError('unexpected full-state input')
    validate_profile(package["profile"])
    validate_update_policy(package["profile"], package["update_policy"])
    if (
        package["profile"]["model"]["seed"] != task["payload"]["seed"]
        or compatibility(package) != task["payload"]["initializer"]["compatibility"]
    ):
        raise ValueError("training seed/compatibility differs")
    if task["purpose"] == "research" and (
        package["scope"] != "NATIVE_VERIFIED" or package["diagnostic_fault"] is not None
    ):
        raise ValueError("diagnostic rows cannot authorize research")
    if (
        package["scope"] not in {"NATIVE_VERIFIED", "DIAGNOSTIC_ONLY"}
        or type(package["refresh_updates"]) is not int
        or package["refresh_updates"] < 1
    ):
        raise ValueError("fixed training scope/refresh boundary required")
    if not isinstance(package["shards"], list) or not 1 <= len(package["shards"]) <= 64:
        raise ValueError("finite shard inventory required")
    fault = package["diagnostic_fault"]
    if fault is not None and (
        not isinstance(fault, dict)
        or set(fault) != {"slot", "update", "point"}
        or fault["point"] != "after_reserve"
        or type(fault["update"]) is not int
        or fault["update"] < 1
        or not isinstance(fault["slot"], str)
    ):
        raise ValueError("closed diagnostic fault coordinate required")
    seen = {}
    for shard in package["shards"]:
        if not isinstance(shard, list) or not 1 <= len(shard) <= 4096:
            raise ValueError("bounded nonempty verified shard required")
        for row in shard:
            fields = {
                "version",
                "identity",
                "ordinal",
                "draw_identity",
                "source_policy",
                "source",
                "structure",
                "behavior",
                "indices",
                "values",
                "target_source",
                "tokens",
                "target_disposition",
                "membership",
                "native_verified",
                "control_events",
                "branch_events",
                "operations",
                "detector_limit",
            }
            if set(row) != fields:
                raise ValueError("closed training row; evaluation metadata excluded")
            if (
                seal(row) != row
                or row.get("version") != "progressive-training-row/v1"
                or row.get("membership") != "training"
            ):
                raise ValueError("sealed training rows only")
            if package["scope"] == "NATIVE_VERIFIED" and row.get("native_verified") is not True:
                raise ValueError("native verification missing")
            if (
                len(row["values"]) != 100
                or len(row["indices"]) != 100
                or encode_body(row["target_source"]) != row["tokens"]
            ):
                raise ValueError("complete target/codec binding required")
            if row["draw_identity"] in seen and seen[row["draw_identity"]] != row:
                raise ValueError("conflicting training row")
            seen[row["draw_identity"]] = row
    probe_ids = {v["id"] for v in package["layout"]["probes"].values()}
    if not probe_ids <= set(seen):
        raise ValueError("complete frozen training probes required")
    Controller(package["layout"], seed=task["payload"]["seed"])
    if "token_target" in task["payload"]:
        from oeis_learn.generalization.progressive.supply import plan
        plan(package,task["payload"]["token_target"],arm=task["payload"]["arm"])
    return package


def _run(
    package,
    authority,
    *,
    accounting,
    control,
    output,
    state_root=Path("/learning-state"),
    predecessor_state=Path("/predecessor-state"),
    initializer_root=None,
):
    import torch

    from oeis_learn.decoder.bigint_codec import BOS_ID, PAD_ID
    from oeis_learn.generalization.progressive.milestones import milestone
    from oeis_learn.pipeline_validation.generalization_curriculum import observe_targets
    from oeis_learn.pipeline_validation.progressive_evidence import publish

    if (
        set(authority) - {"storage"}
        != {"version", "identity", "campaign", "task", "clock", "invocation"}
        or authority["version"] != "progressive-supervisor/v1"
        or seal(authority) != authority
    ):
        raise ValueError("original supervisor authority required")
    c = validate_campaign(authority["campaign"])
    t = validate_task(authority["task"], campaign=c)
    validate_input(package, t)
    from oeis_learn.generalization.progressive.campaign import initializer_origin

    init = t["payload"]["initializer"]
    if init.get("origin") != package.get("origin") and init["artifact"] is not None:
        raise ValueError("initializer origin package differs from frozen task")
    origin, _ = initializer_origin(initializer_root, init, track=t["track"], purpose=t["purpose"])
    from .progressive_device_acceptance import device as execution_device
    device = execution_device(t["purpose"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journal = BudgetJournal(accounting, c["identity"], c["payload"]["caps"])
    from oeis_learn.generalization.progressive.quarantine import Journal

    Journal(journal.root).training(package)
    if t["purpose"] == "research" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("actual HIP required; no CPU fallback")
    lookup = {r["draw_identity"]: r for shard in package["shards"] for r in shard}
    probe_ids = {p["id"] for p in package["layout"]["probes"].values()}

    def row_bin(r):
        return "frontier" if {"loop", "select"} & set(r["operations"]) else "familiar"

    def batch_for(controller):
        b = controller.select()["bin"]
        from oeis_learn.generalization.progressive.supply import shard_index
        index=shard_index(t['payload']['arm'],b,controller.updates,package['refresh_updates'],len(package['shards']))
        rows = [
            r
            for r in package["shards"][index]
            if row_bin(r) == b and r["draw_identity"] not in probe_ids
        ]
        if not rows:
            raise TimeoutError("frozen training bin support deficit")
        return [
            rows[(controller.cursor + i) % len(rows)]
            for i in range(package["update_policy"]["effective_batch"])
        ], index

    progress = Progress(output / "progress")
    status = "PAUSED"
    reason = "UPDATE_BOUNDARY"
    first_failure = None
    with Recovery(
        journal,
        t,
        compatibility(package),
        state_root=state_root,
        checkpoint_bytes=c["payload"]["storage"]["checkpoint_bytes"],
    ) as recovery:
        recovery.attach(authority["clock"])
        if recovery.state["checkpoint"] is None and (
            {"parent_slot", "parent_task"} & set(t["payload"]["initializer"])
        ):
            plans = [
                p
                for p in c["payload"]["tasks"]
                if (
                    p["id"] == t["payload"]["initializer"].get("parent_task")
                    if "parent_task" in t["payload"]["initializer"]
                    else p["slot"] == t["payload"]["initializer"]["parent_slot"]
                )
                and p["arm"] == t["payload"]["arm"]
                and p["seed"] == t["payload"]["seed"]
            ]
            from oeis_learn.generalization.progressive.contracts import record

            parent_plan=plans[0]
            if 'trial_choices' in parent_plan:
                from oeis_learn.generalization.progressive.trial_scopes import resolve
                parent_plan=resolve(parent_plan,t['payload']['trial_choice'])
            parent = record(
                "task",
                dict(parent_plan, campaign=c["identity"]),
                created_at=c["created_at"],
                purpose=c["purpose"],
                track=c["track"],
            )
            with Recovery(
                journal, parent, compatibility(package), state_root=predecessor_state
            ) as previous:
                recovery.initialize_from(previous, campaign=c)
        if recovery.state['checkpoint'] is None and init['mode']=='full_state' and init['artifact'] is not None:
            recovery.initialize_import(initializer_root,package['full_state'])
        was_resumed = recovery.state["checkpoint"] is not None
        worker = RecoverableUpdate(
            recovery, package["profile"], package["update_policy"], device=device
        )
        if (
            worker.context is not None
            and "component_distillation" in package
            and (
                worker.context["selection"].get("component_input")
                != package["component_distillation"]
                or worker.context["selection"].get("component_boundary_task") != t["identity"]
            )
        ):
            worker.adopt_component_cut(package["component_distillation"])
        if worker.context is None:
            mode = t["payload"]["initializer"]["mode"]
            if mode == "weights_only":
                from oeis_learn.generalization.progressive.contracts import verify_refs
                from oeis_learn.pipeline_validation.identity import confined

                if initializer_root is None:
                    raise ValueError("confined initializer root required")
                verify_refs(initializer_root, [package["weights"]], max_bytes=64 << 30)
                weights = torch.load(
                    confined(initializer_root, package["weights"]["path"]),
                    map_location=device,
                    weights_only=True,
                )
                if set(weights) != {"encoder", "decoder"}:
                    raise ValueError("weights-only artifact cannot contain resumable state")
                for name, model in (
                    ("encoder", worker.runner.encoder),
                    ("decoder", worker.runner.decoder),
                ):
                    expected = model.state_dict()
                    if set(weights[name]) != set(expected) or any(
                        not isinstance(v, torch.Tensor)
                        or v.dtype != expected[k].dtype
                        or not bool(torch.isfinite(v).all())
                        for k, v in weights[name].items()
                    ):
                        raise ValueError("incompatible/nonfinite weights-only initializer")
                    model.load_state_dict(weights[name])
            elif mode != "random":
                raise ValueError("unresolved imported initializer")
            controller = Controller(package["layout"], seed=t["payload"]["seed"])
            from .progressive_training import replay_mixer

            replay_input = package.get("component_distillation", package.get("oeis_replay"))
            initial_replay = (
                replay_mixer(replay_input).initial() if replay_input is not None else {}
            )
            context = {
                "scheduler": {"controller": controller.snapshot(), "pre_loss": None},
                "data": {"cursor": 0, "pending_batch": None},
                "replay": initial_replay,
                "history": {},
                "pins": {},
                "selection": {
                    "origin": "new_initialization" if mode == "weights_only" else "fresh_random",
                    "ancestor_cost_available": mode != "weights_only",
                    "initializer_origin": origin["identity"] if origin else None,
                    "prior_exposure": origin["exposure"] if origin else None,
                },
            }
            cuts = {
                k: digest(None)
                for k in (
                    "stream",
                    "allocation",
                    "archive",
                    "library",
                    "cost_journal",
                    "selection",
                    "milestones",
                    "consumed",
                )
            }
            if "oeis_replay" in package:
                cuts["archive"] = context["replay"]["archive"]
            if "component_distillation" in package:
                from .progressive_training import bind_component_cut

                context, cuts = bind_component_cut(context, cuts, package["component_distillation"])
            worker.initialize(context, cuts)
        restored = recovery.restore()
        context = copy.deepcopy(worker.context)
        if "component_distillation" in package:
            from .progressive_training import check_component_cut

            check_component_cut(context, restored["cuts"], package["component_distillation"])
        controller = Controller.restore(context["scheduler"]["controller"], package["layout"])
        missing = restored["discarded_tokens"] - controller.discarded_tokens
        if missing < 0:
            raise ValueError("allocation discarded clock exceeds journal")
        controller.discarded_tokens += missing
        controller.charged_tokens += missing
        context["scheduler"]["controller"] = controller.snapshot()
        pause_file = Path(control) / (
            "pause-" + digest(authority["invocation"]).split(":")[1] + ".json"
        )
        initial = {
            "committed": recovery.state["useful_tokens"],
            "charged": restored["ledger"].consumed,
            "discarded": restored["discarded_tokens"],
            "remaining": journal.task_remaining(t),
            "phase": "STARTUP",
        }
        with Heartbeat(progress, device=device, initial=initial, clock=recovery.clock) as heartbeat:
            try:
                while True:
                    restored = recovery.restore()
                    if restored["ledger"].consumed >= t["payload"]["token_target"]:
                        status='PARTIAL' if recovery.state['pending'] is not None else 'COMPLETE'
                        reason='TARGET_REACHED_WITH_UNCOMMITTED_WORK' if status=='PARTIAL' else 'TOKEN_TARGET_REACHED'
                        break
                    if pause_file.exists():
                        request = load(pause_file)
                        if (
                            seal(request) != request
                            or request.get("campaign") != c["identity"]
                            or request.get("requested") is not True
                        ):
                            raise ValueError("invalid owned pause request")
                        reason = "PAUSE_ACKNOWLEDGED"
                        break
                    remaining_us = recovery.clock["end"] - time.monotonic_ns() // 1000
                    # Reserve peak-live checkpoint bytes from measured frozen storage,
                    # preserving two generations, pinned milestones and package reserve.
                    storage = c["payload"]["storage"]
                    heartbeat.update(phase="CHECKPOINT_RETENTION")
                    recovery.retain(milestone_pins=storage["milestone_pins"])
                    used = sum(p.stat().st_size for p in Path(state_root).rglob("*") if p.is_file())
                    if used + storage["checkpoint_bytes"] > t["payload"]["caps"]["disk_bytes"]:
                        raise TimeoutError("task checkpoint storage cap exhausted")
                    progress.preflight(
                        bytes_required=storage["required_bytes"],
                        inodes_required=storage["required_inodes"],
                        remaining_us=remaining_us,
                        finalization_us=t["payload"]["finalization_us"],
                        research=t["purpose"] == "research",
                        shared_finalization_us=t['payload'].get('trial',{}).get('shared_finalization_us',0),
                        work_us=__import__("oeis_learn.generalization.progressive.iteration",fromlist=["margin"]).margin(t["payload"],research=t["purpose"]=="research"),
                    )
                    if controller.block_position == 8:
                        key = package["layout"]["probes"][controller.pending["bin"]]["id"]
                        probe = lookup[key]
                        measured = observe_targets(
                            worker.runner.encoder,
                            worker.runner.decoder,
                            [dict(probe, values=probe["values"][:20])],
                            device=device,
                        )
                        before = context["scheduler"]["pre_loss"]
                        after = measured["loss_sum"] / measured["tokens"]
                        controller.feedback(
                            role="training",
                            probe=key,
                            gain=max(-1, min(1, (before - after) / max(abs(before), 1))),
                        )
                        context["scheduler"]["pre_loss"] = None
                    rows, index = batch_for(controller)
                    replay_debit = None
                    if "oeis_replay" in package or "component_distillation" in package:
                        from .progressive_training import replay_batch

                        rows, replay_debit = replay_batch(
                            package.get("component_distillation", package.get("oeis_replay")),
                            rows,
                            context["replay"],
                        )
                    if controller.block_position == 0 and context["scheduler"]["pre_loss"] is None:
                        probe = lookup[package["layout"]["probes"][controller.pending["bin"]]["id"]]
                        observed = observe_targets(
                            worker.runner.encoder,
                            worker.runner.decoder,
                            [dict(probe, values=probe["values"][:20])],
                            device=device,
                        )
                        context["scheduler"]["pre_loss"] = observed["loss_sum"] / observed["tokens"]
                    ids = [[BOS_ID, *r["tokens"]] for r in rows]
                    tokens = torch.full(
                        (len(ids), max(map(len, ids))), PAD_ID, dtype=torch.long, device=device
                    )
                    for i, v in enumerate(ids):
                        tokens[i, : len(v)] = torch.tensor(v, device=device)
                    charge = int(tokens[:, 1:].ne(PAD_ID).sum())
                    if charge > journal.task_remaining(t)["tokens"]:
                        status = "PARTIAL"
                        reason = "FULL_UPDATE_CANNOT_FIT"
                        break
                    next_context = copy.deepcopy(context)
                    controller_cut = copy.deepcopy(controller)
                    controller_cut.set_stream_cut(
                        shard=max(index, controller.shard),
                        cursor=controller.cursor + len(rows),
                        rejection_watermark=max(
                            controller.rejection_watermark,
                            max(r["ordinal"] for r in package["shards"][index]) + 1,
                        ),
                    )
                    next_context["scheduler"]["controller"] = controller_cut.snapshot()
                    cuts = dict(
                        restored["cuts"],
                        stream=digest([index, [r["draw_identity"] for r in rows]]),
                        consumed=digest(
                            [restored["cuts"]["consumed"], [r["draw_identity"] for r in rows]]
                        ),
                    )
                    heartbeat.update(
                        phase="UPDATE_IN_PROGRESS",
                        remaining={**journal.task_remaining(t), "elapsed_us": max(0, remaining_us)},
                    )
                    worker.step(
                        [list(map(int, r["values"][:20])) for r in rows],
                        tokens,
                        entries=[r["draw_identity"] for r in rows],
                        update=restored["ledger"].updates + 1,
                        context=next_context,
                        cuts=cuts,
                        boot=authority["clock"]["boot"],
                        monotonic_us=time.monotonic_ns() // 1000,
                        allocation=controller_cut,
                        progress=heartbeat,
                        replay=replay_debit,
                        replay_package=package.get(
                            "component_distillation", package.get("oeis_replay")
                        ),
                        fault_after_reserve=(
                            package["diagnostic_fault"] is not None
                            and package["diagnostic_fault"]["slot"] == t["payload"]["slot"]
                            and package["diagnostic_fault"]["update"]
                            == restored["ledger"].updates + 1
                            and recovery.state["pending"] is None
                        ),
                    )
                    controller = controller_cut
                    context = copy.deepcopy(worker.context)
                    current = recovery.restore()
                    heartbeat.update(
                        committed=recovery.state["useful_tokens"],
                        charged=current["ledger"].consumed,
                        discarded=current["discarded_tokens"],
                        phase="CHECKPOINT_COMMITTED",
                    )
            except (TimeoutError, InterruptedError) as exc:
                reason = type(exc).__name__ + ":" + str(exc)
                status = "PARTIAL" if isinstance(exc, TimeoutError) else "PAUSED"
            except Exception as exc:  # noqa: BLE001 -- preserve the original producer traceback
                import traceback

                from oeis_learn.generalization.progressive.contracts import reference
                from oeis_learn.pipeline_validation.progressive_evidence import safe_failure

                private = output / "private"
                private.mkdir(exist_ok=True)
                private.chmod(0o700)
                trace = private / "train-traceback.txt"
                from oeis_learn.generalization.progressive.storage import writing

                raw_trace = "".join(traceback.format_exception(exc))
                with writing(trace, len(raw_trace.encode())):
                    with trace.open("x") as stream:
                        stream.write(raw_trace)
                trace.chmod(0o600)
                first_failure = safe_failure(
                    exc, producer="installed-train", traceback_ref=reference(output, trace)
                )
                status = "FAILED"
                reason = "LEARNER_FAILURE"
            final = recovery.restore()
            heartbeat.update(
                phase=status,
                committed=recovery.state["useful_tokens"],
                charged=final["ledger"].consumed,
                discarded=final["discarded_tokens"],
                remaining=journal.task_remaining(t),
            )
        recovery.finish_external(status)
        current = recovery.restore()
        elapsed = max(0, time.monotonic_ns() // 1000 - authority["clock"]["start_us"])
        m = milestone(
            current["ledger"].consumed,
            ancestor_tokens=recovery.base["tokens"],
            ancestor_cost_us=recovery.base["ancestor_cost_us"],
            new_cost_us=elapsed,
        )
        from oeis_learn.generalization.progressive.diversity import summarize
        committed={}
        for path in sorted(recovery.root.glob('event-*.json')):
            event=load(path)['state']
            last=event['last']
            if last is not None:committed[last['update']]=last['batch']['entries']
        from collections import Counter
        consumed=Counter(i for entries in committed.values() for i in entries if i in lookup)
        diversity=summarize(list(lookup.values()),consumed)
        atomic_write(output/'diversity.json',seal(dict(version='progressive-consumed-diversity/v1',task=t['identity'],checkpoint=recovery.state['checkpoint']['identity'],counts=diversity,excluded_nonmechanical_entries=sum(i not in lookup for entries in committed.values() for i in entries))))
        atomic_write(output / "checkpoint.json", recovery.state["checkpoint"])
        atomic_write(output / "milestone.json", m)
        return publish(
            output,
            operation="train",
            mechanical_status="COMPLETE"
            if status == "COMPLETE"
            else "FAILED"
            if status == "FAILED"
            else "PARTIAL",
            reason=reason,
            producer_error=first_failure,
            campaign=c["identity"],
            task=t["identity"],
            stage=t["payload"]["stage"],
            metrics={
                "task_status": status,
                "checkpoint": recovery.state["checkpoint"]["identity"],
                "checkpoint_record":__import__("oeis_learn.generalization.progressive.contracts",fromlist=["reference"]).reference(output,output/"checkpoint.json"),
                "weights":__import__("oeis_learn.generalization.progressive.contracts",fromlist=["reference"]).reference(output,output/"checkpoint.json"),
                "diversity":__import__('oeis_learn.generalization.progressive.contracts',fromlist=['reference']).reference(output,output/'diversity.json'),
                "charged_tokens": current["ledger"].consumed,
                "useful_tokens": recovery.state["useful_tokens"],
                "discarded_tokens": current["discarded_tokens"],
                "milestone": m["identity"],
                "device": device,
                "diagnostic_only": t["purpose"] == "acceptance",
                "initialization": "new_initialization"
                if t["payload"]["initializer"]["mode"] == "weights_only"
                else "resume"
                if was_resumed
                else "fresh_random",
                "ancestor_cost_available": context["selection"].get(
                    "ancestor_cost_available", True
                ),
            },
        )


def run(
    package,
    authority,
    *,
    accounting,
    control,
    output,
    state_root=Path("/learning-state"),
    predecessor_state=Path("/predecessor-state"),
    initializer_root=None,
):
    from oeis_learn.generalization.progressive.storage import owned_guard

    validate_task(authority["task"], campaign=authority["campaign"])
    # Preflight precedes the very first checkpoint and includes previous owned
    # invocation outputs supplied by the supervisor's append-only inventory.
    with owned_guard(authority, output=output, state_root=state_root).activate():
        return _run(
            package,
            authority,
            accounting=accounting,
            control=control,
            output=output,
            state_root=state_root,
            predecessor_state=predecessor_state,
            initializer_root=initializer_root,
        )
