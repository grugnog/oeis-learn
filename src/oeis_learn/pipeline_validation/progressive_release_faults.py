"""Terminate a disposable worker inside backward and resume its actual saved update."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    atomic_write,
    digest,
    load,
    seal,
)
from .progressive_panel_sources import require


def child(sample, output, device):
    from .progressive_generator_probe import measured_updates

    def interrupt(runner, checkpoint, contract, cap, weights):
        # Autograd calls this only once the second batch is inside a real backward pass.
        def stop(grad):
            if device == "cuda":
                import torch

                torch.cuda.synchronize()
            atomic_write(
                output / "interrupted.json",
                seal(
                    dict(
                        checkpoint=checkpoint,
                        contract=contract,
                        cap=cap,
                        weights=weights,
                        useful_updates=1,
                        discarded_updates=1,
                    )
                ),
            )
            os.kill(os.getpid(), signal.SIGKILL)

        next(p for p in runner.decoder.parameters() if p.requires_grad).register_hook(stop)

    measured_updates(
        sample["profile"],
        sample["batches"],
        "ordinary",
        output / "checkpoint",
        device=device,
        microbatch=sample["microbatch"],
        after_checkpoint=interrupt,
    )
    raise ValueError("backward interruption was never observed")


def resume(sample, output, device):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.decoder.bigint_codec import PAD_ID
    from .progressive_generator_probe import batch
    from .progressive_training import PolicyUpdate, RecoverableUpdate, resolve_update_policy

    if device == "cuda":
        from .progressive_qualification import require_hip

        require_hip()
    cut = load(output / "interrupted.json")
    require(cut == seal(cut), "partial update changed")
    state = GeneralizationCheckpointStore(output / "checkpoint", cut["contract"], cut["cap"]).load(
        cut["checkpoint"]
    )
    values = state["states"]
    policy = resolve_update_policy(
        sample["profile"],
        objective="ordinary",
        effective_batch=32,
        microbatch_size=sample["microbatch"],
        schedule_horizon_tokens=4_000_000,
        weight_decay=0,
    )
    runner = PolicyUpdate(sample["profile"], policy, device=device)
    runner.encoder.load_state_dict(values["weights"]["encoder"])
    runner.decoder.load_state_dict(values["weights"]["decoder"])
    runner.load_state_dict(values["optimizer"])
    RecoverableUpdate._rng_restore(values["rng"])
    require(
        model_snapshot(runner.encoder, runner.decoder) == cut["weights"]
        and values["data"]["cursor"] == 32
        and values["data"]["pending_batch"] is None
        and values["scheduler"]["completed_updates"] == 1
        and values["selection"]["consumed"]
        == digest([s["positive"]["draw_identity"] for s in sample["batches"][0]]),
        "restored optimizer boundary or seed stream differs",
    )
    seq, tokens = batch([s["positive"] for s in sample["batches"][1]], device)
    n = int(tokens[:, 1:].ne(PAD_ID).sum())
    _, ledger = runner.update(seq, tokens, ledger=state["ledger"].reserve("1", n))
    require(ledger.consumed == cut["cap"], "resumed token accounting differs")
    return dict(
        final=model_snapshot(runner.encoder, runner.decoder),
        useful_updates=2,
        discarded_updates=1,
        useful_tokens=ledger.consumed,
    )


def learner_fault(output, sample_path, expected, *, device="cuda"):
    import tempfile

    sample = load(sample_path, maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(sample == seal(sample), "disposable learner sample differs")
    with tempfile.TemporaryDirectory(dir=output) as stage:
        target = Path(stage)
        start = time.monotonic_ns()
        # Use a fresh interpreter, including a fresh HIP runtime.
        proc = subprocess.run(
            [
                sys.executable,
                *(["-I"] if device == "cuda" else []),
                "-m",
                __name__,
                str(sample_path),
                stage,
                device,
            ],
            check=False,
        )
        failed_us = (time.monotonic_ns() - start) // 1000
        require(
            proc.returncode == -signal.SIGKILL and (target / "interrupted.json").is_file(),
            "disposable learner did not stop during backward",
        )
        start = time.monotonic_ns()
        result = resume(sample, target, device)
        require(result["final"] == expected, "resumed update differs from uninterrupted reference")
        return dict(result, failed_us=failed_us, restored_us=(time.monotonic_ns() - start) // 1000)


if __name__ == "__main__":
    sample_path, destination, device = sys.argv[1:]
    child(load(sample_path, maximum=CAMPAIGN_DOCUMENT_BYTES), Path(destination), device)
