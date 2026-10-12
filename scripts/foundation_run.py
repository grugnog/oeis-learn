#!/usr/bin/env python3
"""Fresh diagnostics from the VM: bulk data, preparation, private training and analysis."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def configuration(path):
    import yaml

    from oeis_learn.experiments.profiles import qualified_model_profile

    value = yaml.safe_load(Path(path).read_text())
    if (
        set(value) != {"version", "data", "cohort", "archive", "sampler_config", "run"}
        or value["version"] != "fresh-workflow/v1"
    ):
        raise ValueError("closed fresh-workflow/v1 configuration required")
    if set(value["data"]) != {
        "cache",
        "revision",
        "seed",
        "max_records",
        "max_download_bytes",
        "free_disk_floor_bytes",
    }:
        raise ValueError("unknown/missing bulk configuration")
    schemas = {
        "cohort": {"seed", "dev_count", "final_count"},
        "archive": {"training_prompts", "development_structures", "seconds"},
        "run": {"seed", "training_seconds", "evaluation"},
    }
    for section, keys in schemas.items():
        if set(value[section]) != keys:
            raise ValueError(f"unknown/missing {section} configuration")
    if set(value["run"]["evaluation"]) != {"groups", "attempts", "tokens", "per_group_seconds"}:
        raise ValueError("closed evaluation configuration required")
    for section in ("data", "cohort", "archive", "run"):
        for name, number in value[section].items():
            if name in {"cache", "revision", "evaluation"}:
                continue
            if type(number) is not int or number < 0:
                raise ValueError(f"exact nonnegative integer required: {section}.{name}")
    if any(type(n) is not int for n in value["run"]["evaluation"].values()):
        raise ValueError("exact evaluation integers required")
    if any(not 0 <= value[s]["seed"] < 2**32 for s in ("data", "cohort", "run")):
        raise ValueError("seed must be in0..2**32-1")
    if (
        not 1 <= value["data"]["max_records"] <= 10000
        or not 0 < value["data"]["max_download_bytes"] <= 100 << 30
    ):
        raise ValueError("bulk record/byte cap exceeded")
    if (
        not 128 <= value["archive"]["training_prompts"] <= 10000
        or not 1 <= value["archive"]["development_structures"] <= 1000
        or not 1 <= value["archive"]["seconds"] <= 7200
    ):
        raise ValueError("bounded archive targets required")
    if (
        not 1 <= value["cohort"]["dev_count"] <= 1000
        or not 1 <= value["cohort"]["final_count"] <= 1000
    ):
        raise ValueError("nonempty bounded development/final cohorts required")
    if not 60 <= value["run"]["training_seconds"] <= 1800:
        raise ValueError("first sustained pilot training target must be 60..1800 seconds")
    evaluation = value["run"]["evaluation"]
    if (
        not 1 <= evaluation["attempts"] <= 32
        or not 8 <= evaluation["tokens"] <= 1023
        or not 1 <= evaluation["groups"] <= 16
        or not 1 <= evaluation["per_group_seconds"] <= 60
    ):
        raise ValueError("bounded first-pilot evaluation required")
    value["run"]["model"] = qualified_model_profile(attention_backend="eager")
    value["sampler_config"] = str((ROOT / value["sampler_config"]).resolve())
    return value


def relative(path):
    target = Path(path).resolve()
    try:
        return str(target.relative_to(ROOT))
    except ValueError:
        raise ValueError("input/output must be in the shared checkout") from None


def remote(*args):
    copied = ROOT / "system/lab-gpu.py"
    client = [sys.executable, str(copied), "client"] if copied.is_file() else ["lab-gpu"]
    return subprocess.run([*client, "--cwd", ".", "foundation", "--", *args], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data.add_argument("operation", choices=["fetch"])
    data.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prepare.add_argument("--output", default="tmp-fresh-inputs")
    check = commands.add_parser("check")
    check.add_argument("--inputs", required=True)
    for operation in ("train", "smoke"):
        command = commands.add_parser(operation)
        command.add_argument("--inputs", required=True)
        command.add_argument("--run", required=True)
        if operation == "train":
            command.add_argument("--profile", default="fresh-pilot-1h", choices=["fresh-pilot-1h"])
    for operation in ("resume", "status", "cancel", "evaluate", "finalize", "pack"):
        command = commands.add_parser(operation)
        command.add_argument("--run", required=True)
        if operation == "evaluate":
            command.add_argument("--split", choices=["development", "final"], default="development")
            command.add_argument("--checkpoint", choices=["latest"], default="latest")
        if operation == "finalize":
            command.add_argument(
                "--checkpoint", choices=["best-development"], default="best-development"
            )
        if operation == "pack":
            command.add_argument("--output", required=True)
            command.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in {"data", "prepare"}:
            config = configuration(ROOT / args.config)
            cache = ROOT / relative(ROOT / config["data"]["cache"])
            if args.command == "data":
                from oeis_learn.fresh.bulk import fetch

                result = fetch({k: v for k, v in config["data"].items() if k != "cache"}, cache)
                print(
                    json.dumps(
                        {
                            "snapshot": result["snapshot"],
                            "files": len(result["selected"]),
                            "bytes": result["bytes"],
                        },
                        indent=2,
                    )
                )
            else:

                result = prepare(config, cache, ROOT / relative(args.output))
                print(
                    json.dumps(
                        {"identity": result["identity"], "output": args.output, "qualified": False},
                        indent=2,
                    )
                )
        elif args.command == "pack":
            from oeis_learn.fresh.bundle import pack

            print(
                json.dumps(
                    pack(ROOT / relative(args.run), ROOT / args.output, checkpoint=args.checkpoint),
                    indent=2,
                )
            )
        elif args.command == "check":
            from oeis_learn.fresh.inputs import check_inputs

            print("INPUTS PASS", check_inputs(ROOT / relative(args.inputs))["identity"], flush=True)
            remote("check")
        elif args.command in {"train", "smoke"}:
            from oeis_learn.fresh.inputs import check_inputs

            check_inputs(ROOT / relative(args.inputs))
            try:
                remote(
                    "run",
                    relative(args.inputs),
                    relative(args.run),
                    "smoke" if args.command == "smoke" else args.profile,
                )
            finally:
                if (ROOT / args.run / "run.json").exists():
                    from oeis_learn.fresh.bundle import pack

                    output = ROOT / "tmp-analysis" / (Path(args.run).name + ".zip")
                    print(json.dumps(pack(ROOT / args.run, output), indent=2))
        else:
            extra = (
                [args.split, args.checkpoint]
                if args.command == "evaluate"
                else [args.checkpoint] if args.command == "finalize" else []
            )
            remote(args.command, relative(args.run), *extra)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
