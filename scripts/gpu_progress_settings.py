"""Discover candidate real inputs for the VM GPU progression runner.

No downloads, checkpoint conversion, archive admission or qualification claims.
Only generated settings/cohort/case files are written in tmp-gpu-inputs.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def reject_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def discover(root):
    out = root / "tmp-gpu-inputs"
    if out.is_symlink():
        raise ValueError("Input output directory must not be a symlink")
    out.mkdir(exist_ok=True)
    sources, holdouts, archives = [], {}, []
    excluded = {"tests", "target", "build", "dist", "node_modules", "__pycache__", "venv"}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith((".", "tmp-gpu-progress"))
            and d not in excluded
            and not (Path(directory) / d).is_symlink()
            and not (Path(directory) / d / "receipts.json").is_file()
        )
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue
            path = Path(directory) / filename
            if path.is_symlink():
                continue
            try:
                value = json.loads(path.read_text(), parse_constant=reject_constant)
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            version = value.get("version")
            if version == "indexed-source/v1" and value.get("records"):
                sources.append(str(path))
            elif version == "expanded-holdouts/v1" and (
                set(value)
                == {"version", "seed", "structures", "compositions", "parameter_bits", "prefixes"}
                and type(value["seed"]) is int
                and all(
                    isinstance(value[k], list) for k in ("structures", "compositions", "prefixes")
                )
            ):
                payload = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
                key = "sha256:" + hashlib.sha256(payload).hexdigest()
                holdouts.setdefault(key, (str(path), value))
            elif (
                version == "program-archive/v1"
                and filename == "index.json"
                and isinstance(value.get("holdouts"), str)
            ):
                archives.append((str(path.parent), value.get("holdouts")))

    pairs = [(archive, *holdouts[key]) for archive, key in archives if key in holdouts]

    def single(label, candidates):
        display = [c[:2] if isinstance(c, tuple) else c for c in candidates]
        print(label + ":", json.dumps(display, indent=2))
        return candidates[0] if len(candidates) == 1 else None

    source = single("source manifest candidates", sources)
    pair = single("matching archive/holdouts candidates", pairs)
    seed = pair[2]["seed"] if pair else 20260913
    cohort = out / "cohort.auto.yaml"
    if cohort.is_symlink():
        raise ValueError("Cohort output must not be a symlink")
    cohort.write_text(
        "schema_version: foundation/v1\n"
        "profile: prefix20_total100_affine_v2\n"
        f"seed: {seed}\ndev_count: 2\nfinal_count: 2\n"
    )
    cases = []
    for config in sorted(root.glob("runs/*/config.yaml")):
        candidates = []
        for checkpoint in config.parent.rglob("model_epoch_*.v2.pt"):
            match = re.fullmatch(r"model_epoch_(\d+)\.v2\.pt", checkpoint.name)
            if match and int(match[1]) > 0 and checkpoint.is_file():
                candidates.append((int(match[1]), str(checkpoint)))
        if candidates:
            cases.append({"checkpoint": max(candidates)[1], "config": str(config)})
    legacy = out / "legacy-cases.auto.json"
    if cases:
        if legacy.is_symlink():
            raise ValueError("Legacy case output must not be a symlink")
        legacy.write_text(json.dumps(cases, indent=2) + "\n")
    print("historical checkpoint candidates:", json.dumps(cases, indent=2))
    settings = dict(
        image_id=None,
        preflight_config=None,
        source_manifest=source,
        cohort_config=str(cohort),
        archive=pair[0] if pair else None,
        holdouts=pair[1] if pair else None,
        legacy_cases=str(legacy) if cases else None,
        final_seconds=3600,
    )
    destination = out / "settings.auto.json"
    if destination.is_symlink():
        raise ValueError("Settings output must not be a symlink")
    destination.write_text(json.dumps(settings, indent=2) + "\n")
    print("WROTE", destination)
    missing = [
        key
        for key in ("source_manifest", "archive", "holdouts", "legacy_cases")
        if not settings[key]
    ]
    print("MISSING/AMBIGUOUS:", missing)
    print("Discovered candidates are not qualified; gpu_progress.py validates their actual bytes.")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    discover(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
