"""Private HIP benchmark of actual FP32 SFT, with no checkpoint or truth input."""

import gc
import math
import threading
import time
from pathlib import Path

from .common import digest, read, write
from .tuning import choose_trial, permutation


def hardware_sample():
    from oeis_learn.tracking.foundation_metrics import sensor_metrics

    result = {"utc_ns": time.time_ns(), "sensors": sensor_metrics("cuda"), "gpu_busy_percent": None}
    paths = list(Path("/sys/class/drm").glob("card[0-9]*/device/gpu_busy_percent"))
    if len(paths) == 1:
        try:
            value = int(paths[0].read_text())
            if 0 <= value <= 100:
                result["gpu_busy_percent"] = value
        except (OSError, ValueError):
            pass
    return result


def training_step(
    encoder, decoder, optimizer, rows, order, cursor, size, generator, buckets, *, device="cuda"
):
    import torch

    from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
    from oeis_learn.rl.sft_trainer import supervised_step

    if cursor + size > len(order):
        order, cursor = permutation(rows, generator, size, buckets), 0
    batch = [rows[i] for i in order[cursor : cursor + size]]
    cursor += len(batch)
    tokens = [[BOS_ID] + row["tokens"] for row in batch]
    tensor = torch.full(
        (len(batch), max(map(len, tokens))), PAD_ID, device=device, dtype=torch.long
    )
    for i, ids in enumerate(tokens):
        tensor[i, : len(ids)] = torch.tensor(ids, device=device)
    result = supervised_step(
        encoder,
        decoder,
        optimizer,
        [[int(v) for v in row["values"]] for row in batch],
        tensor,
        device=device,
        pad_id=PAD_ID,
    )
    return (
        result,
        sum(len(row["tokens"]) for row in batch),
        tensor.numel() - len(batch),
        order,
        cursor,
    )


def benchmark(inputs, output):
    import torch

    from .worker import models, runtime

    cfg, examples = read(inputs / "settings.json"), read(inputs / "examples.json")
    if digest(examples) != cfg["examples_sha256"] or cfg["device"] != "cuda":
        raise ValueError("HIP benchmark requires frozen admitted training examples")
    rows = examples["training"]
    if len(rows) < 128:
        raise ValueError("benchmark needs at least 128 admitted prompts")
    trials = []
    for size in (4, 8, 16, 32, 64):
        for bucketed in (False, True):
            encoder = decoder = optimizer = None
            stop, samples = threading.Event(), []

            def observe(stop=stop, samples=samples):
                while not stop.is_set():
                    samples.append(hardware_sample())
                    stop.wait(1)

            observer = threading.Thread(target=observe, daemon=True)
            trial = {"batch_size": size, "length_buckets": bucketed, "status": "OOM"}
            try:
                encoder, decoder, _ = models(cfg)
                encoder.train()
                decoder.train()
                optimizer = torch.optim.AdamW(
                    [*encoder.parameters(), *decoder.parameters()],
                    lr=cfg["learning_rate"],
                    foreach=False,
                )
                generator = torch.Generator().manual_seed(cfg["seed"] + 1)
                order = permutation(rows, generator, size, bucketed)
                cursor = 0

                # Warm kernels and also stress the longest observed training shape.
                torch.cuda.reset_peak_memory_stats()
                order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]), reverse=True)
                _, _, _, order, cursor = training_step(
                    encoder, decoder, optimizer, rows, order, cursor, size, generator, bucketed
                )
                order, cursor = permutation(rows, generator, size, bucketed), 0
                for _ in range(3):
                    _, _, _, order, cursor = training_step(
                        encoder, decoder, optimizer, rows, order, cursor, size, generator, bucketed
                    )
                stress_peak = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                observer.start()
                start, tokens, padded, steps, losses = time.monotonic(), 0, 0, 0, []
                while time.monotonic() - start < 20:
                    result, count, capacity, order, cursor = training_step(
                        encoder, decoder, optimizer, rows, order, cursor, size, generator, bucketed
                    )
                    losses.append(result["loss"])
                    tokens += count
                    padded += capacity
                    steps += 1
                elapsed = time.monotonic() - start
                trial.update(
                    status="PASS",
                    steps=steps,
                    seconds=elapsed,
                    tokens_per_second=tokens / elapsed,
                    padding_efficiency=tokens / padded,
                    peak_allocated_bytes=max(stress_peak, torch.cuda.max_memory_allocated()),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    finite_losses=all(math.isfinite(v) for v in losses),
                )
                if not trial["finite_losses"]:
                    raise FloatingPointError("benchmark loss is nonfinite")
            except torch.cuda.OutOfMemoryError as exc:
                trial["error"] = str(exc)
            finally:
                stop.set()
                if observer.is_alive():
                    observer.join(timeout=2)
                trial["hardware_samples"] = samples
                encoder = decoder = optimizer = None
                gc.collect()
                torch.cuda.empty_cache()
            trials.append(trial)
            write(output / "progress.json", {"trials": trials})
            print(f"BENCH batch={size} buckets={bucketed}: {trial['status']}", flush=True)
    winner = choose_trial(trials)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "version": "fresh-benchmark/v1",
        "trials": trials,
        "recommended": {
            k: winner[k] for k in ("batch_size", "length_buckets", "tokens_per_second")
        },
        "device": "cuda:0",
        "device_name": properties.name,
        "architecture": getattr(properties, "gcnArchName", None),
        "device_memory_bytes": properties.total_memory,
        "runtime": runtime("cuda"),
        "qualified": False,
        "selection": "within5percent of fastest, smaller batch; <=48GiB peak",
    }
