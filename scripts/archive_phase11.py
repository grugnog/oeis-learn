"""Archive/replay actual CI evidence in the qualification runtime, never infer GPU readiness."""

import argparse
from pathlib import Path
import shutil

from oeis_learn.evaluation.handoff import archive_handoff, replay_handoff
from oeis_learn.experiments.artifacts import canonical_bytes, compute_file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    root = Path(args.evidence).resolve()
    disposition = Path("specs/007-experiment-foundation/validation/phase-11-dispositions.json")
    shutil.copyfile(disposition, root / "dispositions.json")
    wheels = list(root.glob("installed-wheel-*/installed-wheel.xml"))
    if len(wheels) != 1:
        raise ValueError("exactly one installed-wheel acceptance report required")
    readiness = dict(
        g7="g7.json",
        g8="learning/g8.json",
        g9="discovery/g9.json",
        wheel_junit=wheels[0].relative_to(root).as_posix(),
        phase10_junit="phase10.xml",
        role_junit="roles.xml",
        dispositions="dispositions.json",
    )
    # Deliberate roots only; installed environments/wheels and previous archives
    # are excluded. Gate validators replay the copied JUnit/native evidence.
    paths = {p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".xml", ".txt")}
    paths.add(wheels[0])
    for name in ("native", "learning", "discovery"):
        paths.update(p for p in (root / name).rglob("*") if p.is_file())
    paths.discard(root / "handoff-inputs.json")
    manifest = dict(
        version="handoff-inputs/v1",
        files=[
            dict(path=p.relative_to(root).as_posix(), sha256=compute_file_hash(p))
            for p in sorted(paths)
        ],
        readiness=readiness,
    )
    (root / "handoff-inputs.json").write_bytes(canonical_bytes(manifest))
    report = archive_handoff(manifest, input_root=root, output=root / "handoff")
    assert replay_handoff(root / "handoff") == report
    gates = report["readiness"]["gates"]
    assert gates["G7"]["status"] == gates["G9"]["status"] == "PASS", gates
    assert gates["G8"]["status"] == "pending" and gates["G8"]["software_status"] == "PASS", gates
    assert all(
        gates["G10"]["checks"][name]["status"] == "PASS"
        for name in ("installed_wheel_native", "integration", "dispositions")
    ), gates
    assert not report["readiness"]["full_007_complete"] and not report["archive_is_authorization"]
    print(canonical_bytes(report).decode())


if __name__ == "__main__":
    main()
