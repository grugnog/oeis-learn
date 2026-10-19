#!/usr/bin/env python3
"""Image-side CPU tasks; the public host CLI is foundation_run.py."""

import argparse
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("OEIS_WORKSPACE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def configuration(path):
    import yaml

    from oeis_learn.experiments.profiles import qualified_model_profile

    value = yaml.safe_load(Path(path).read_text())
    if isinstance(value, dict) and value.get("version") == "bigint-bootstrap/v1":
        from oeis_learn.fresh.bigint_config import resolve, packaged_artifacts

        return resolve(value, artifacts=packaged_artifacts())
    if set(value) != {"version", "data", "cohort", "archive", "sampler_config", "run"} or value[
        "version"
    ] not in {"fresh-workflow/v1", "fresh-workflow/v2"}:
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
        optional = (
            (
                {"profile", "batch_size", "length_buckets", "encoder", "learning"}
                if value["version"] == "fresh-workflow/v2"
                else {"profile", "batch_size", "length_buckets"}
            )
            if section == "run"
            else {"sampling_shards"}
            if section == "archive"
            else set()
        )
        if not keys <= set(value[section]) or set(value[section]) - keys - optional:
            raise ValueError(f"unknown/missing {section} configuration")
    if set(value["run"]["evaluation"]) != {"groups", "attempts", "tokens", "per_group_seconds"}:
        raise ValueError("closed evaluation configuration required")
    for section in ("data", "cohort", "archive", "run"):
        for name, number in value[section].items():
            if name in {
                "cache",
                "revision",
                "evaluation",
                "profile",
                "length_buckets",
                "encoder",
                "learning",
            }:
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
        or not 1 <= value["archive"].get("sampling_shards", 1) <= 64
    ):
        raise ValueError("bounded archive targets required")
    if (
        not 1 <= value["cohort"]["dev_count"] <= 1000
        or not 1 <= value["cohort"]["final_count"] <= 1000
    ):
        raise ValueError("nonempty bounded development/final cohorts required")
    profile = value["run"].get("profile", "fresh-pilot-1h")
    if profile not in {"fresh-pilot-1h", "fresh-extended-12h"}:
        raise ValueError("unknown training profile")
    maximum = 36000 if profile == "fresh-extended-12h" else 1800
    if not 60 <= value["run"]["training_seconds"] <= maximum:
        raise ValueError("training target exceeds the selected profile")
    if value["run"].get("batch_size", 4) not in {4, 8, 16, 32, 64}:
        raise ValueError("batch size must be one of 4, 8, 16, 32, 64")
    if type(value["run"].get("length_buckets", False)) is not bool:
        raise ValueError("length_buckets must be an explicit boolean")
    evaluation = value["run"]["evaluation"]
    if (
        not 1 <= evaluation["attempts"] <= 32
        or not 8 <= evaluation["tokens"] <= 1023
        or not 1 <= evaluation["groups"] <= 16
        or not 1 <= evaluation["per_group_seconds"] <= 60
    ):
        raise ValueError("bounded first-pilot evaluation required")
    if value["version"] == "fresh-workflow/v2":
        from oeis_learn.experiments.bootstrap_models import bootstrap_profile
        from oeis_learn.fresh.bootstrap_config import validate_learning

        value["run"]["model"] = bootstrap_profile(value["run"].get("encoder"))
        value["run"]["learning"] = validate_learning(value["run"].get("learning"))
        if (
            value["run"]["learning"]["curriculum"]
            and value["archive"]["development_structures"] < 4
        ):
            raise ValueError("curriculum requires at least four development structures")
    else:
        value["run"]["model"] = qualified_model_profile(attention_backend="eager")
    value["sampler_config"] = str((ROOT / value["sampler_config"]).resolve())
    return value


def relative(path):
    target = Path(path).resolve()
    try:
        return str(target.relative_to(ROOT))
    except ValueError:
        raise ValueError("input/output must be in the shared checkout") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data.add_argument("operation", choices=["fetch"])
    data.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prepare.add_argument("--output", default="tmp-fresh-inputs")
    seal = commands.add_parser("seal-inputs")
    seal.add_argument("--inputs", required=True)
    seal.add_argument("--accept-shortfall", action="store_true", required=True)
    check = commands.add_parser("check")
    check.add_argument("--inputs", required=True)
    check.add_argument(
        "--inputs-only", action="store_true", help="validate inputs without launching GPU probes"
    )
    resolved = commands.add_parser("bigint-resolve")
    resolved.add_argument("--config", required=True)
    resolved.add_argument("--output", required=True)
    resolved.add_argument("--seed", type=int)
    compare = commands.add_parser("bigint-compare")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "bigint-resolve":
            from oeis_learn.fresh.bigint_config import load, packaged_artifacts
            from oeis_learn.fresh.common import write

            write(
                ROOT / relative(args.output),
                load(ROOT / args.config, seed=args.seed, artifacts=packaged_artifacts()),
            )
        elif args.command == "bigint-compare":
            from oeis_learn.fresh.comparison import compare_runs

            print(
                json.dumps(
                    compare_runs(
                        [ROOT / relative(p) for p in args.runs], ROOT / relative(args.output)
                    ),
                    indent=2,
                )
            )
        elif args.command == "seal-inputs":
            from oeis_learn.fresh.input_sealing import seal

            result = seal(ROOT / relative(args.inputs))
            print(
                json.dumps(
                    {
                        "identity": result["identity"],
                        "completion": result.get("admission_completion"),
                        "qualified": False,
                    },
                    indent=2,
                )
            )
        elif args.command in {"data", "prepare"}:
            config = configuration(ROOT / args.config)
            cache = ROOT / relative(ROOT / config["data"]["cache"])
            if args.command == "data":
                from oeis_learn.fresh.bulk import fetch

                result = fetch(
                    {
                        k: v
                        for k, v in config["data"].items()
                        if k
                        in {
                            "revision",
                            "seed",
                            "max_records",
                            "max_download_bytes",
                            "free_disk_floor_bytes",
                        }
                    },
                    cache,
                )
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
                from oeis_learn.fresh.inputs import prepare as prepare_inputs

                result = prepare_inputs(config, cache, ROOT / relative(args.output))
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

        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, OSError) and exc.errno in {errno.EMFILE, errno.ENFILE}:
            import resource
            import traceback

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"Open-file limit: soft={soft}, hard={hard}. Traceback follows:", file=sys.stderr)
            traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
