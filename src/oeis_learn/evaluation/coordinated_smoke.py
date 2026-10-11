"""Host-enforced strict smoke using only content-bound generic pool capabilities."""

from pathlib import Path
from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    load_json,
    canonical_bytes,
    atomic_write,
)
from oeis_learn.tracking.role_coordinator import RoleCoordinator
from oeis_learn.evaluation.live_measurement import write_view
from oeis_learn.evaluation.foundation_cohort import read_ref


def pool_capabilities(root):
    root = Path(root)
    manifest = load_json((root / "manifest.json").read_bytes())
    if manifest.get("kind") != "generic_pool":
        raise ValueError("strict generic pool required")
    files = {"pool/manifest.json": root / "manifest.json"}

    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                name = "pool/" + value["path"]
                if name not in files:
                    content = read_ref(root, value)
                    files[name] = ArtifactPath(root, value["path"]).as_path()
                    visit(content)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    # Canonical fixed trainer entrypoints are independently validated in the role.
    for name in ("manifest.json", "examples.json"):
        files["pool/trainer/" + name] = root / "trainer" / name
    return files


def coordinated_smoke(config, pool, image, output, *, seconds=600, resume_run=None, run_provenance=None, stop_after=None):
    if type(seconds) is not int or not 0 < seconds <= 600:
        raise ValueError("bounded600-second smoke required")
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= 3):
        raise ValueError("stop_after must be a completed update boundary in 1..3")
    if run_provenance is not None and (not isinstance(run_provenance, dict) or run_provenance.get("final_image_id") != image):
        raise ValueError("run provenance must bind the selected role image")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    files = {"config.yaml": Path(config)}
    if resume_run is None:
        files.update(pool_capabilities(pool))
    else:
        # Already separated trainer/checkpoint/run journals; never a cohort root.
        previous = Path(resume_run)
        if not (previous / "run.json").exists():
            raise ValueError("complete previous smoke run required")
        if Path(config).read_bytes() != (previous / "checkpoints/config.yaml").read_bytes():
            raise ValueError("resume cannot override the frozen smoke configuration")
        if run_provenance is not None:
            prior = load_json((previous / "checkpoints/contract.json").read_bytes())["run_provenance"]
            if any(prior.get(key) != value for key, value in run_provenance.items()):
                raise ValueError("resume cannot override frozen run provenance")
        for directory in ("trainer", "checkpoints", "budget-events", "events"):
            for file in (previous / directory).rglob("*"):
                if file.is_file():
                    files["run/" + str(file.relative_to(previous))] = file
        files["run/run.json"] = previous / "run.json"
    inputs = write_view(
        root / "input",
        {"settings.json": {"device": "cuda", "seconds": seconds, "resume": resume_run is not None,
                           "run_provenance": run_provenance, "stop_after": stop_after}},
        files,
    )
    coordinator = RoleCoordinator(root / "coordination", image, workers=4)
    try:
        result = coordinator.run(
            "smoke", role="learner", inputs=inputs, output=root / "smoke", seconds=seconds, gpu=True
        )
    finally:
        coordinator.close()
    events = list(coordinator.log.events.values())
    report = dict(
        version="coordinated-smoke/v1",
        image=image,
        result=result,
        roles=events,
        main_memory_bytes=88 << 30,
        worker_memory_bytes=1 << 30,
        workers=4,
        host_floor_bytes=16 << 30,
        aggregate_memory_bytes=92 << 30,
        diagnostic_small_host=False,
        final_truth_available=False,
    )
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    report["report_sha256"] = digest(report)
    atomic_write(ArtifactPath(root, "coordinated-smoke.json"), canonical_bytes(report))
    return report
