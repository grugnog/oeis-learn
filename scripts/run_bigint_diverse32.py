#!/usr/bin/env python3
"""Run one preregistered diverse32 diagnostic after reviewing its scoped HIP receipt.

Host Python is stdlib-only. Reuses the reviewed image, prepared inputs and cost
measurements. Always archives logs/results; never starts a transfer screen.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def reviewed_recipe(report, *, image, source_sha256, shape_sha256):
    from oeis_learn.fresh.common import digest
    from oeis_learn.fresh.cost_policy import validate_preflight

    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != digest(payload):
        raise ValueError("reviewed benchmark hash mismatch")
    neural = report.get("neural", {})
    if (
        report.get("status") != "PASS"
        or not neural.get("hip")
        or not neural.get("device", "").startswith("cuda")
        or neural.get("measurement_protocol") != "fixed-state-batched/v2"
        or neural.get("generation_parity") is not True
        or not neural.get("frozen_states")
        or not report.get("fitting", {}).get("complete")
    ):
        raise ValueError("complete actual HIP, fixed-state and parity evidence required")
    receipt = report.get("cost_receipts", {}).get("diverse32", {})
    sealed = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != digest(sealed) or receipt.get("status") != "PASS":
        raise ValueError("reviewed passing diverse32 receipt required")
    recipe = receipt["measured_recipe"]["recipe"]
    if recipe.get("identity") != digest({k: v for k, v in recipe.items() if k != "identity"}):
        raise ValueError("admitted recipe hash mismatch")
    cost = recipe["learning"]["cost_preflight"]
    if (
        report.get("source_sha256") != source_sha256
        or report.get("image") != image
        or receipt.get("scope") != "diverse32"
        or receipt.get("shape_sha256") != shape_sha256
        or report.get("costs") != cost["costs"]
        or neural.get("shape_sha256") != shape_sha256
    ):
        raise ValueError("reviewed source/image/shape/cost mismatch")
    if recipe["model"]["seed"] != 20261016:
        raise ValueError("T096 registers exactly seed20261016")
    forecast = validate_preflight(
        recipe,
        scope="diverse32",
        image=image,
        source_sha256=source_sha256,
        shape_sha256=shape_sha256,
    )
    if receipt.get("projection") != forecast:
        raise ValueError("reviewed projection differs from executable schedule")
    return recipe, forecast


def claim_run(root, recipe, run):
    """An interrupted attempt keeps its claim and original ledger; no fresh retry."""
    from oeis_learn.fresh.common import digest, lock, read, write

    cost = recipe["learning"]["cost_preflight"]
    key = digest(
        {
            k: cost[k]
            for k in ("scope", "source_sha256", "shape_sha256", "profile_sha256", "schedule_sha256")
        }
    )[7:]
    path = Path(root) / "run-state/phase5-diverse32" / (key + ".json")
    with lock(path.with_suffix(".lock")):
        if path.exists():
            old = read(path)
            raise ValueError(
                "diverse32 attempt already claimed at "
                + old["run"]
                + "; inspect it and resume its original ledger, not a new run"
            )
        write(
            path,
            dict(run=str(run), recipe_sha256=recipe["identity"], status="started"),
            immutable=True,
        )
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("auto", "docker", "podman"), default="auto")
    p.add_argument(
        "--reviewed-preflight", required=True, help="reviewed performance/benchmark.json"
    )
    p.add_argument("--inputs", required=True)
    p.add_argument(
        "--output", required=True, help="fresh output directory; ZIP is written beside it"
    )
    p.add_argument("--image")
    p.add_argument("--gpu-base-image")
    p.add_argument("--render-node")
    p.add_argument("--memory-gib", type=float)
    p.add_argument("--cpus", type=int)
    p.add_argument("--free-disk-gib", type=int, default=5)
    p.add_argument("--host-free-memory-gib", type=float, default=1)
    args = p.parse_args()
    from foundation_local import Host, relative
    from qualify_bigint_phase5 import archive, launch, sha, shape_manifest, write
    from oeis_learn.fresh.common import digest
    from oeis_learn.experiments.source_snapshot import collect_snapshot

    output = ROOT / relative(args.output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("new output directory required")
    output.mkdir(parents=True)
    summary = dict(
        version="phase5-diverse32/v1",
        status="FAIL",
        qualified=False,
        scope="diverse32",
        pending=["T089", "T090", "T091", "T093", "T094", "T095", "T096", "T097", "T098"],
    )
    code = 1
    try:
        source = ROOT / relative(args.reviewed_preflight)
        report = json.loads(source.read_text())
        args.image = args.image or report["image"]
        host = Host(args)
        image = host.ensure()
        inputs = ROOT / relative(args.inputs)
        recipe, projection = reviewed_recipe(
            report,
            image=image,
            source_sha256=digest(collect_snapshot(ROOT)),
            shape_sha256=digest(shape_manifest(inputs)),
        )
        public = [
            sys.executable,
            str(ROOT / "scripts/foundation_run.py"),
            "--engine",
            host.engine.name,
            "--image",
            image,
        ]
        launch(
            [*public, "check", "--inputs", relative(str(inputs))],
            output / "check-command",
            seconds=600,
        )
        write(output / "reviewed-benchmark.json", report)
        write(output / "measured-recipe.yaml", recipe)  # JSON is YAML; host needs no PyYAML.
        run = output / "learnability-run"
        claim = claim_run(ROOT, recipe, relative(str(run)))
        summary.update(
            image=image,
            source_sha256=report["source_sha256"],
            reviewed_preflight_sha256=sha(source),
            projection=projection,
            recipe_sha256=recipe["identity"],
            claim=relative(str(claim)),
            run=relative(str(run)),
        )
        write(output / "summary.json", summary)
        launch(
            [
                *public,
                "learnability",
                "--config",
                relative(str(output / "measured-recipe.yaml")),
                "--inputs",
                relative(str(inputs)),
                "--view",
                "diverse",
                "--seed",
                "20261016",
                "--run",
                relative(str(run)),
            ],
            output / "learnability-command",
            seconds=recipe["budgets"]["training_seconds"] + 300,
        )
        launch(
            [
                *public,
                "pack",
                "--run",
                relative(str(run)),
                "--output",
                relative(str(output / "learnability.zip")),
            ],
            output / "pack-command",
            seconds=300,
        )
        result = json.loads((run / "summary.json").read_text())
        summary.update(
            result=result,
            checkpoint=json.loads((run / "checkpoint-identity.json").read_text()),
            budget=json.loads((run / "budget.json").read_text()),
            status="PASS" if result.get("memorization_gate") is True else "BLOCKED",
        )
        summary["pending"].remove("T096")
        code = 0 if summary["status"] == "PASS" else 2
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write(output / "summary.json", summary)
        target = archive(output)
        print(f"Results: {output / 'summary.json'} Archive: {target}", flush=True)
    return code


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
