"""Installed CPU/HIP roles for the registered generator pilot. Inputs are file-only projections."""

import os
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_pilot import ARMS, LOOKS
from .progressive_panel_sources import require

REQUEST = "round028-pilot-role/v1"
OUTPUT = "round028-pilot-output/v1"


def prepare_rows(data, client, output):
    from oeis_learn.generalization.progressive.generator import default_policy, draw
    from .progressive_preparation import verify_draw

    law = default_policy(seed=data["seed"])
    start, count = data["start"], data["count"]
    require(
        type(start) is int
        and type(count) is int
        and count > 0
        and 0 <= start < start + count <= law["max_draws"],
        "registered mechanical counter range required",
    )
    rows, rejected = [], {}
    for ordinal in range(start, start + count):
        row, reason = verify_draw(draw(law, ordinal), client=client)
        if row is None:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            rows.append(row)
    path = output / "rows.json"
    atomic_write(
        path,
        seal(
            dict(
                version="round028-pilot-rows/v1",
                seed=data["seed"],
                start=start,
                count=count,
                rows=rows,
                rejected=rejected,
            )
        ),
    )
    return {"rows": reference(output, path)}


def restore_inputs(root, checkpoint, contract, cap):
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore

    previous = GeneralizationCheckpointStore(root, contract, cap)
    return previous.load(checkpoint)


def train(work, data, *, device="cuda"):
    import torch
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.training_rows import (
        matched_batch,
        matched_milestone,
        exposure_at,
    )
    from .preparation_recovery import revise
    from .progressive_generator_probe import batch
    from .progressive_training import PolicyUpdate, RecoverableUpdate, resolve_update_policy
    from .progressive_research_inputs import checked_file

    if device == "cuda":
        from .progressive_qualification import require_hip

        require_hip()
    package = load(checked_file(work, data["training"]), maximum=CAMPAIGN_DOCUMENT_BYTES)
    stream = package["stream"]
    arm, target = data["arm"], data["target"]
    require(
        seal(package) == package
        and package["identity"] == data["package"]
        and seal(stream) == stream
        and stream["registration"] == data["registration"]
        and stream["pair"] == 0
        and arm in ARMS
        and target in LOOKS
        and package["profile"]["model"]["seed"] == stream["seeds"]["initialization"],
        "registered matched stream required",
    )
    rows = {r["draw_identity"]: r for r in package["rows"]}
    labels = {v["positive_row"]: v["label"] for v in stream["pairs"]}
    require(
        set(stream["order"]) <= rows.keys() and bool(labels),
        "eligible frozen training labels required",
    )
    policy = resolve_update_policy(
        package["profile"],
        objective=arm,
        effective_batch=32,
        microbatch_size=package["microbatch"],
        weight_decay=0,
        schedule_horizon_tokens=LOOKS[-1],
    )
    contract = digest([package["identity"], arm, policy["identity"]])
    cap = LOOKS[-1] + 32 * max(len(v["tokens"]) for v in rows.values())
    worker = PolicyUpdate(package["profile"], policy, device=device)
    initial = model_snapshot(worker.encoder, worker.decoder)
    selection = {"initial_weights": initial, "matched_stream": stream["identity"]}
    previous = data["previous"]
    ledger = Ledger(cap)
    consumed = digest(None)
    if previous is not None:
        state = restore_inputs(work / "inputs/previous", previous, contract, cap)
        ledger = state["ledger"]
        saved = state["states"]
        require(
            saved["selection"] == selection
            and saved["data"]["cursor"] == ledger.updates
            and saved["history"]["look"] == LOOKS[LOOKS.index(target) - 1],
            "full predecessor checkpoint required",
        )
        worker.encoder.load_state_dict(saved["weights"]["encoder"])
        worker.decoder.load_state_dict(saved["weights"]["decoder"])
        worker.load_state_dict(saved["optimizer"])
        RecoverableUpdate._rng_restore(saved["rng"])
        consumed = saved["history"]["order"]
        del saved, state
    elif target != 0:
        raise ValueError("missing predecessor at a positive-token look")
    output = work / "output"
    debits = BudgetJournal(
        output / "accounting", contract, dict(dict.fromkeys(KEYS, 0), tokens=cap)
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    while ledger.consumed < target:
        keys = matched_batch(stream, ledger.updates * 32, 32)
        samples = [rows[k] for k in keys]
        sequences, tokens = batch(samples, device)
        positive = sum(len(r["tokens"]) for r in samples)
        key = str(ledger.updates + 1)
        debit = debits.reserve(
            arm,
            "update",
            ledger.updates + 1,
            dict(dict.fromkeys(KEYS, 0), tokens=positive),
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
        )
        require(not debit["committed"], "repeated model update without its restored checkpoint")
        _, ledger = worker.update(
            sequences,
            tokens,
            ledger=ledger.reserve(key, positive),
            preference_labels=[labels.get(k) for k in keys] if arm == ARMS[1] else None,
        )
        debits.commit(debit["key"])
        consumed = digest([consumed, keys])
        if ledger.updates % 32 == 0:
            revise(
                output / "progress.json",
                {"updates": ledger.updates, "positive_tokens": ledger.consumed},
            )
    if device == "cuda":
        torch.cuda.synchronize()
    expected = exposure_at(stream, rows, ledger.updates)
    require(
        expected["order_digest"] == consumed and expected["useful_tokens"] == ledger.consumed,
        "unequal ordered training exposure",
    )
    # Both the existing full-state store and the matched milestone independently reopen this cut.
    context = {
        "selection": selection,
        "scheduler": {"controller": {"cursor": 32 * ledger.updates}},
        "data": {"cursor": ledger.updates, "pending_batch": None},
        "replay": {},
        "pins": {},
        "history": {"look": target, "order": consumed, "predecessor": previous},
    }
    store = GeneralizationCheckpointStore(output / "checkpoint", contract, cap)
    ref = store.save(RecoverableUpdate.capture_runner(worker, context), ledger)
    full = store.load(ref)
    full.update(cuts={"selection": digest(selection), "consumed": consumed}, discarded_tokens=0)
    cut = matched_milestone(
        dict(matched_generator=stream, profile=package["profile"], update_policy=policy),
        rows,
        {"identity": ref["sha256"]},
        full,
        ledger.consumed,
        target,
    )
    manifest = load(output / "checkpoint" / ref["path"])
    return dict(
        milestone=cut,
        checkpoint=ref,
        weights=manifest["states"]["weights"],
        contract=contract,
        cap=cap,
        updates=ledger.updates,
    )


def evaluate_role(work, data, client, role):
    from .pilot_clocks import CORRECT_FULL, DESCRIPTIVE
    from .progressive_evaluation import generate, score
    from .progressive_qualification import require_hip

    if role == "search":
        require_hip()
        policy = CORRECT_FULL if data["target"] == LOOKS[-1] else DESCRIPTIVE
        checkpoint = data["weights"]
        path = generate(
            work,
            "generation",
            data["rows"],
            data["recipe"],
            client=client,
            model_control="retained_D" if data["arm"] == "incumbent" else "earlier_checkpoint",
            reference_recipe=None if data["arm"] == "incumbent" else data["recipe"],
            device="cuda",
            checkpoint=checkpoint,
            methods=("correct",),
            evaluation_policy=policy,
            seconds=data["seconds"],
            output_root=work / "output",
            case_boundary_dir=work / "output" / "case-boundaries",
        )
        return {"candidates": path}
    score(
        work,
        data["candidates"],
        data["rows"],
        client=client,
        output="output/scored",
        seconds=data["seconds"],
        selected_only=True,
    )
    return {"scores": reference(work / "output", work / "output/scored/scores.json")}


def main():
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.synthesis.client import SynthesisClient
    from .progressive_generator_probe import usage_summary
    from oeis_learn.tracking.pipeline_profile import Recorder

    work = Path("/work")
    req = load(work / "inputs/request.json")
    data = load(work / "inputs/data.json", maximum=CAMPAIGN_DOCUMENT_BYTES)
    allowed = {
        "prepare": {"seed", "start", "count"},
        "train": {"arm", "target", "previous", "training", "package", "registration"},
        "search": {"arm", "target", "rows", "recipe", "weights", "seconds"},
        "score": {"rows", "candidates", "seconds"},
    }
    role = req["role"]
    require(
        registered_request(req, data, allowed)
        and req["source"] == digest(source_snapshot())
        and req["image"] == os.environ["OEIS_QUALIFICATION_IMAGE"]
        and req["engine"] == os.environ["OEIS_QUALIFICATION_ENGINE"],
        "wrong pilot role, source, image or input projection",
    )
    start = time.monotonic_ns()
    output = work / "output"
    with SynthesisClient(digest([REQUEST, req["identity"]])) as client:
        native = client.runtime_identity("bigint-body/v1")
        require(req["candidate_native"] in (None, native), "candidate native identity changed")
        if role in ("train", "search"):
            nodes = sorted(Path("/dev/dri").glob("renderD*"))
            with (
                Recorder(
                    output / "telemetry",
                    instrumented=True,
                    render_node=nodes[0] if len(nodes) == 1 else None,
                    host_counters=False,
                ) as recorder,
                recorder.phase(role),
            ):
                result = (
                    train(work, data)
                    if role == "train"
                    else evaluate_role(work, data, client, role)
                )
            result["telemetry"] = usage_summary(
                output / "telemetry", (time.monotonic_ns() - start) // 1000
            )
        elif role == "prepare":
            result = prepare_rows(data, client, output)
        else:
            result = evaluate_role(work, data, client, role)
    atomic_write(
        output / "producer.json",
        seal(
            dict(
                version=OUTPUT,
                request=req["identity"],
                native=native,
                result=result,
                elapsed_us=(time.monotonic_ns() - start) // 1000,
            )
        ),
    )


def registered_request(req, data, allowed):
    role = req.get("role")
    return (
        req.get("version") == REQUEST
        and seal(req) == req
        and seal(data) == data
        and req["data"] == data["identity"]
        and role in allowed
        and set(data) == allowed[role] | {"block", "identity"}
    )


if __name__ == "__main__":
    main()
