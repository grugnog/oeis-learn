"""Fixed owning checks for the disposable VM progression driver; lazy imports."""

from pathlib import Path
import importlib.metadata
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time
from gpu_progress import (
    ROOT,
    HOST_STEPS,
    ISOLATED_ROLES_BLOCKER,
    Blocked,
    read,
    write,
    check_junit,
    task_inventory,
)


def command(argv, **kwargs):
    print("+ " + json.dumps([str(a) for a in argv]), flush=True)
    subprocess.run([str(a) for a in argv], cwd=kwargs.pop("cwd", ROOT), check=True, **kwargs)


def pytest(out, nodes, *, name="tests", allowed_skips=(), extra=(), env=None):
    path = out / (name + ".xml")
    command(
        [sys.executable, "-m", "pytest", "-q", *nodes, *extra, "--junitxml=" + str(path)],
        env=env,
    )
    return dict(tests=check_junit(path, allowed_skips), junit=str(path))


def archive(cfg):
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.experiments.learning_profiles import load_holdouts

    if not cfg.get("archive") or not cfg.get("holdouts"):
        raise Blocked("Actual strict archive and exact holdouts required")
    return ProgramArchive(cfg["archive"], holdouts=load_holdouts(read(cfg["holdouts"])))


def host_capability(cfg, out):
    raise Blocked(ISOLATED_ROLES_BLOCKER)


def ensure_image(cfg):
    """The broker prints only the immutable content ID to stdout, logs to stderr."""
    result = subprocess.run(
        ["lab-gpu", "image", "ensure"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    actual = result.stdout.strip()
    if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", actual):
        raise ValueError("lab-gpu image ensure did not return one immutable image ID")
    actual = "sha256:" + actual.removeprefix("sha256:")
    if cfg.get("image_id") and cfg["image_id"] != actual:
        raise Blocked("Configured image_id differs from lab-gpu image ensure")
    return dict(runtime=dict(image_id=actual, image_inputs=read(ROOT / "gpu-project.json")))


def require_sampling_strata(strata):
    if not strata["model_samples"] or not strata["verified_solutions"]:
        raise Blocked("No valid actual model/verified stratum; inspect retained sampling failures")


def preflight_config(cfg, out):
    if cfg.get("preflight_config"):
        return Path(cfg["preflight_config"])
    import re

    image = re.search(r"^ARG GPU_BASE_IMAGE=(.+)$", (ROOT / "Dockerfile").read_text(), re.M).group(
        1
    )
    base, sep, digest = image.partition("@")
    if not sep:
        raise ValueError("Pinned root GPU base required")
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    for name in ("Dockerfile", "uv.lock"):
        shutil.copyfile(ROOT / name, inputs / name)
    config = dict(
        schema_version="preflight/v1",
        kind="preflight_config",
        base_image=base,
        base_image_digest=digest,
        final_image_id=cfg["image_id"],
        dockerfile="Dockerfile",
        dependency_lock="uv.lock",
        requested_device="hip",
        dtype="float32",
        no_cpu_fallback=True,
        no_driver_install=True,
        no_privileged_container=True,
        updates=3,
        optimizer="adamw",
        visible_terms=20,
        body_tokens=1024,
        timeout_minutes=10,
        fixtures=[
            dict(kind=k)
            for k in ("zero", "alternating_signed256_extremes", "adjacent_large_integer_prefixes")
        ],
        model=dict(
            d_model=256,
            n_heads=4,
            n_encoder_layers=4,
            n_decoder_layers=4,
            d_ff=1024,
            dropout=0.1,
            max_seq_len=1024,
            use_film=True,
            enable_summary_tokens=False,
        ),
    )
    path = inputs / "config.json"
    write(path, config)
    return path


def provenance(cfg, state):
    from oeis_learn.evaluation.experiment_runner import implementation_identity
    from oeis_learn.experiments.artifacts import (
        compute_file_hash,
        compute_canonical_digest as digest,
    )

    report = read(state / "preflight/preflight-report.json")
    if report["status"] != "ready" or report["final_image_id"] != cfg["image_id"]:
        raise ValueError("Ready image-bound preflight required")
    return dict(
        source_revision=implementation_identity(),
        dependency_lock=compute_file_hash(ROOT / "uv.lock"),
        base_image_digest=report["base_image_digest"],
        final_image_id=cfg["image_id"],
        environment_report=digest(report["environment_report"]),
    )


def check_broker_client():
    for binary in ("lab-gpu", "ssh"):
        if not shutil.which(binary):
            raise Blocked(f"{binary} client unavailable; rebuild the root-Dockerfile VM")
    if not os.environ.get("LAB_GPU_USER") or not os.environ.get("LAB_GPU_IDENTITY"):
        raise Blocked("LAB_GPU_USER and LAB_GPU_IDENTITY must be set by Smolfile")
    key = Path(os.environ["LAB_GPU_IDENTITY"])
    if not key.is_file() or not os.access(key, os.R_OK):
        raise Blocked("LAB_GPU_IDENTITY must name a readable dedicated SSH key")
    try:
        ROOT.relative_to("/workspace")
    except ValueError:
        raise Blocked("The VM checkout must be mounted under /workspace for lab-gpu")


def inventory(out):
    versions = {
        n: importlib.metadata.version(n)
        for n in ("torch", "pytest", "wasmtime", "numpy", "oeis-learn")
    }
    runtime = dict(python=sys.version, platform=platform.platform(), packages=versions)
    write(
        out / "inventory.json",
        dict(
            runtime=runtime,
            tasks=task_inventory(),
            disk=shutil.disk_usage(ROOT)._asdict(),
            lab_gpu=shutil.which("lab-gpu"),
        ),
    )
    if not Path("/proc/sys/kernel/random/boot_id").is_file():
        raise Blocked("Linux boot identity unavailable; run inside the root-Dockerfile VM")
    check_broker_client()
    from oeis_learn.sandbox.pipeline import runtime_identity

    runtime["native"] = runtime_identity("rust")
    runtime["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return dict(runtime=runtime)


def cpu_regressions(out):
    paths = sorted(
        str(p.relative_to(ROOT))
        for area in ("unit", "contract", "integration")
        for p in (ROOT / "tests" / area).glob("test_foundation*.py")
        if p.name not in {"test_foundation_learning_gpu.py", "test_foundation_gpu.py"}
    )
    nodes = paths + [
        "tests/integration/test_foundation_gpu.py::test_unavailable_gpu_never_falls_back"
    ]
    skips = (
        "test_actual_docker_role_views_and_reclamation",
        "test_gpu_checkpoint_loading_and_prefix_only_generation",
        "test_complete_effective_constructors_and_real_forward_backward[cuda]",
        "test_installed_train_resume_search",
        "test_installed_discover_and_fresh_certificate",
        "test_actual_docker_worker_mount_and_device_isolation",
        "test_legacy_train",
        "test_legacy_resume",
        "test_legacy_warmstart",
        "test_legacy_evaluate",
    )
    return pytest(
        out, nodes, name="cpu", allowed_skips=skips, extra=("-m", "not gpu and not docker")
    )


def gpu_environment(out, cfg):
    import torch
    from oeis_learn.cli.foundation_preflight import inspect_host, host_supports_rocm
    from oeis_learn.sandbox.pipeline import runtime_identity

    host = inspect_host()
    if not host_supports_rocm(host) or not torch.version.hip or not torch.cuda.is_available():
        raise Blocked("Real HIP device required; no CPU fallback")
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    if not torch.equal(y, torch.full_like(y, 128)):
        raise ValueError("HIP matrix multiplication mismatch")
    runtime = dict(
        image_id=cfg["image_id"],
        torch=str(torch.__version__),
        hip=torch.version.hip,
        device_name=torch.cuda.get_device_name(0),
        native=runtime_identity("rust"),
    )
    return dict(
        runtime=runtime, observed_utc=time.time(), host=host, disk=shutil.disk_usage(ROOT)._asdict()
    )


def baseline(cfg, state, out):
    from oeis_learn.evaluation.repair_gate import run_qualification, require_g7
    from oeis_learn.data.program_pool import build_pool
    from oeis_learn.evaluation.coordinated_smoke import coordinated_smoke
    from oeis_learn.evaluation.baseline_gate import SUITES, assemble_baseline_bundle

    host_capability(cfg, out)
    # Keep compiler artifacts outside source identity and stage evidence. The
    # shared cache is writable even when the image's /opt build tree is not.
    os.environ["CARGO_TARGET_DIR"] = str(state / "cargo-cache")
    g7 = run_qualification(out / "repairs")
    require_g7(g7, evidence_root=out / "repairs")
    build_pool(
        ROOT / "configs/foundation/generic_sampler.yaml",
        state / "release-source/cohort",
        out / "pool",
    )
    smoke = coordinated_smoke(
        ROOT / "configs/foundation/wat_smoke.yaml",
        out / "pool",
        cfg["image_id"],
        out / "coordinated",
        run_provenance=provenance(cfg, state),
    )
    run = out / "coordinated/smoke/run"
    checkpoint = run / "checkpoints" / smoke["result"]["checkpoint_name"]
    refs = {}
    for gate, modules in SUITES.items():
        paths = []
        for module in sorted(modules):
            choices = list((ROOT / "tests").glob("*/" + module + ".py"))
            if len(choices) != 1:
                raise ValueError("One owning module required: " + module)
            paths.append(str(choices[0].relative_to(ROOT)))
        pytest(out, paths, name=gate, allowed_skips=("test_unavailable_gpu_never_falls_back",))
        refs[gate] = out / (gate + ".xml")
    refs["native_parity"] = out / "repairs/native/parity.json"
    refs["containment"] = out / "coordinated/coordinated-smoke.json"
    assemble_baseline_bundle(
        checkpoint,
        run,
        state / "preflight/preflight-report.json",
        refs,
        out / "bundle",
        preflight_config=read(state / "preflight/config-reference.json")["path"],
    )
    selection = dict(checkpoint=str(checkpoint), run=str(run))
    write(out / "selected.json", selection)
    return selection


def legacy(cfg, out, device):
    import torch
    from oeis_learn.evaluation.checkpoint import load_checkpoint_v2
    from oeis_learn.evaluation.legacy_wrappers import load_legacy_configuration
    from oeis_learn.experiments.artifacts import (
        compute_file_hash,
        compute_canonical_digest as digest,
    )

    path = Path(cfg["legacy_cases"])
    cases = read(path)
    if not isinstance(cases, list) or not cases:
        raise Blocked("Actual historical parent/configuration cases required")
    reports = []
    for index, case in enumerate(cases):
        if (
            not isinstance(case, dict)
            or set(case) - {"checkpoint", "config", "tests"}
            or not {"checkpoint", "config"} <= set(case)
        ):
            raise ValueError("Closed legacy case required")
        parent = (path.parent / case["checkpoint"]).resolve()
        config = (path.parent / case["config"]).resolve()
        historical = load_legacy_configuration(config)
        e, d, identity = load_checkpoint_v2(str(parent), device=torch.device("cpu"))
        if identity.epoch <= 0:
            raise ValueError("Actual trained historical epoch required")
        e.eval()
        d.eval()
        tokens = torch.tensor([[0, 1, 2, 3]], device="cpu")
        with torch.no_grad():
            expected = d(tokens, e.forward_from_sequences([[7] * 20], device="cpu")).cpu()
        if device == "cuda":
            if not torch.version.hip or not torch.cuda.is_available():
                raise Blocked("HIP required for historical GPU checks")
            e.to(device)
            d.to(device)
            with torch.no_grad():
                actual = d(
                    tokens.to(device), e.forward_from_sequences([[7] * 20], device=device)
                ).cpu()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        del e, d
        directory = out / str(index)
        directory.mkdir()
        os.environ.update(
            OEIS_LEGACY_PARENT=str(parent),
            OEIS_LEGACY_CONFIG=str(config),
            OEIS_LEGACY_OUTPUT=str(directory),
            OEIS_LEGACY_DEVICE=device,
        )
        tests = case.get("tests")
        if tests is not None and (
            not isinstance(tests, dict)
            or set(tests) != {"train", "resume", "warmstart", "evaluate"}
        ):
            raise ValueError("All four owning legacy operations required")
        for op in ("train", "resume", "warmstart", "evaluate"):
            nodes = (
                tests[op]
                if tests
                else ["tests/integration/test_foundation_legacy_artifacts.py::test_legacy_" + op]
            )
            if (
                not isinstance(nodes, list)
                or not nodes
                or any(not isinstance(n, str) for n in nodes)
            ):
                raise ValueError("Nonempty owning pytest node IDs required")
            pytest(directory, nodes, name=op)
        if device == "cuda":
            from oeis_learn.cli.train_warmstart import run_warmstart_pipeline

            gate = run_warmstart_pipeline(
                str(parent), str(directory / "gate"), stage="gate", dry_run=True
            )
            write(directory / "warmstart-gate.json", gate)
        reports.append(
            dict(
                parent_sha256=compute_file_hash(parent),
                config_sha256=compute_file_hash(config),
                effective_configuration_sha256=digest(historical),
                epoch=identity.epoch,
                device=device,
                purpose="legacy_unverified",
                scope="bounded shared paths",
                full_optimizer_rng_resume=False,
                sustained_historical_launcher_acceptance="pending",
            )
        )
    write(out / "legacy.json", reports)
    return dict(cases=reports)


def overflow(cfg, state, out):
    import torch
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.evaluation.search import ModelProposer, SearchSession
    from oeis_learn.tracking.role_coordinator import DockerRunner
    from oeis_learn.evaluation.overflow_diagnostics import (
        STRATA,
        archive_strata,
        verify_origins,
        diagnose_overflow,
        score_development,
    )
    from oeis_learn.evaluation.experiment_runner import implementation_identity
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
    from oeis_learn.tracking.foundation_metrics import sensor_metrics

    host_capability(cfg, out)
    a = archive(cfg)
    snapshot = a.freeze()
    cpu = read(state / "phase13-inputs/overflow/overflow.json")
    strata = read(state / "phase13-inputs/overflow/strata.json")
    if cpu["code_sha256"] != implementation_identity() or cpu["indices"] != list(range(100)):
        raise ValueError("CPU trace identity mismatch")
    packets = read(state / "phase13-inputs/overflow/provenance.json")
    selection = read(state / "baseline/selected.json")
    cp = load_foundation_checkpoint(selection["checkpoint"], device="cuda", include_payload=True)
    proposer = ModelProposer(cp.encoder, cp.decoder, context_limit=1024)
    rows = list(a.training_view(snapshot))
    if not rows:
        raise Blocked("Training-only visible prefixes required")
    runner = DockerRunner(out / "workers", cfg["image_id"], workers=4)
    draws = []
    torch.cuda.synchronize()
    before = sensor_metrics("cuda")
    start = time.perf_counter_ns()
    try:
        for number in range(32):
            values = list(map(int, rows[number % len(rows)][1]["entry"]["conditioning_values"]))
            session = SearchSession(values, portfolio="greedy", seed=7001 + number)
            result = session.step(proposer, runner)
            write(out / "searches" / f"{number}.json", result)
            hypothesis = result.get("provenance", {}).get("hypothesis") if result else None
            if hypothesis:
                draws.append(
                    (
                        result["source"],
                        dict(search=hypothesis, conditioning=[str(v) for v in values]),
                    )
                )
    finally:
        runner.close()
    torch.cuda.synchronize()
    telemetry = dict(
        before=before,
        after=sensor_metrics("cuda"),
        sampling_ns=time.perf_counter_ns() - start,
        allocated=torch.cuda.memory_allocated(),
        peak_allocated=torch.cuda.max_memory_allocated(),
    )
    write(out / "telemetry.json", telemetry)
    del proposer, cp
    torch.cuda.empty_cache()
    combined = archive_strata(a, snapshot)
    combined["random_generic"] = strata["random_generic"]
    for source, packet in draws:
        key = digest(packet)
        packets[key] = packet
        combined["model_samples"].append(
            dict(source=source, origin=STRATA["model_samples"], provenance_sha256=key)
        )
    for name in STRATA:
        combined[name] = list({row["source"]: row for row in combined[name]}.values())[:256]
    require_sampling_strata(combined)
    origin = verify_origins(combined, a, snapshot, provenance_packets=packets)
    cache = {
        (r["source"], r["provenance_sha256"]): r
        for s in cpu["strata"].values()
        for r in s["records"]
    }
    outstanding = {
        name: [r for r in rows if (r["source"], r["provenance_sha256"]) not in cache]
        for name, rows in combined.items()
    }
    fresh = diagnose_overflow(
        outstanding, out / "outstanding", archive=a, snapshot=snapshot, provenance_packets=packets
    )
    cache.update(
        {
            (r["source"], r["provenance_sha256"]): r
            for s in fresh["strata"].values()
            for r in s["records"]
        }
    )
    report = dict(cpu, source_sha256=digest(combined), provenance=origin, strata={})
    for name, rows in combined.items():
        records = [cache[(r["source"], r["provenance_sha256"])] for r in rows]
        report["strata"][name] = dict(
            availability="available" if rows else "unavailable",
            population=len(rows),
            records=records,
            shortfall=256 - len(rows),
        )
    rescued = {
        r["source_sha256"]
        for s in report["strata"].values()
        for r in s["records"]
        if r["wide"].get("outcome") is None
        and r["wide"].get("violations", 0) > 0
        and r["wide"].get("final_outputs_in_range")
    }
    report["rescued_candidates"] = len(rescued)
    write(out / "strata.json", combined)
    write(out / "provenance.json", packets)
    write(out / "seal.json", dict(strata_sha256=digest(combined), provenance=origin))
    report.update(score_development(report, state / "release-source/cohort", out))
    report["report_sha256"] = digest(report, "report_sha256")
    write(out / "overflow.json", report)
    return dict(
        report_sha256=report["report_sha256"],
        telemetry=telemetry,
        reused_records=sum(len(rows) for rows in combined.values())
        - sum(len(rows) for rows in outstanding.values()),
    )


def measurement(cfg, state, out):
    from oeis_learn.evaluation.measurement_inputs import prepare_measurement
    from oeis_learn.evaluation.experiment_runner import measure

    host_capability(cfg, out)
    selection = read(state / "baseline/selected.json")
    manifest = prepare_measurement(
        archive(cfg),
        state / "release-source/cohort",
        state / "baseline/repairs/g7.json",
        state / "gpu-learning/g8.json",
        state / "baseline/repairs",
        cfg["image_id"],
        out / "input",
        baseline_report=state / "baseline/bundle/baseline.json",
        baseline_checkpoint=selection["checkpoint"],
    )
    report = measure(manifest, out / "run", source_root=out / "input")
    if not report.get("qualified"):
        raise ValueError("Engineering/pilot did not qualify; preserve immutable actual disposition")
    return report


def clean_wheel(out):
    """Install the project wheel first, then expose the image's locked dependencies."""
    import site
    import tarfile
    import zipfile

    root = out / "wheel"
    root.mkdir()
    command(
        ["uv", "build", "--offline", "--no-build-isolation", "--wheel", "--out-dir", root / "dist"]
    )
    command(["uv", "venv", "--python", sys.executable, root / "env"])
    python = root / "env/bin/python"
    wheels = list((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Exactly one clean project wheel required")
    command(["uv", "pip", "install", "--python", python, "--no-deps", "--offline", wheels[0]])
    packages = list((root / "env/lib").glob("python*/site-packages"))
    if len(packages) != 1:
        raise ValueError("One clean environment required")
    (packages[0] / "oeis-image-dependencies.pth").write_text(
        "\n".join(site.getsitepackages()) + "\n"
    )
    acceptance = root / "acceptance"
    acceptance.mkdir()
    shutil.copyfile(
        ROOT / "tests/contract/test_foundation_qualified_cli.py",
        acceptance / "test_installed_cli.py",
    )
    env = dict(os.environ, OEIS_INSTALLED_WHEEL="1")
    env.pop("PYTHONPATH", None)
    command(
        ["uv", "build", "--offline", "--no-build-isolation", "--sdist", "--out-dir", root / "dist"]
    )
    source = root / "source"
    source.mkdir()
    with tarfile.open(next((root / "dist").glob("*.tar.gz"))) as bundle:
        bundle.extractall(source, filter="data")
    sources = [p for p in source.iterdir() if p.is_dir()]
    if len(sources) != 1:
        raise ValueError("One source distribution required")
    command(
        [
            "uv",
            "build",
            "--offline",
            "--no-build-isolation",
            "--wheel",
            "--out-dir",
            root / "rebuilt",
            sources[0],
        ]
    )

    def resources(path):
        with zipfile.ZipFile(path) as bundle:
            return {n: bundle.read(n) for n in bundle.namelist() if n.startswith("oeis_learn/")}

    if resources(wheels[0]) != resources(next((root / "rebuilt").glob("*.whl"))):
        raise ValueError("Source rebuild changed qualification resources")
    command(
        [
            python,
            "-I",
            "-m",
            "pytest",
            "-q",
            "tests/unit/test_foundation_task_inventory.py",
            "tests/unit/test_foundation_handoff.py",
            "--junitxml=" + str(root / "source-distribution.xml"),
        ],
        cwd=sources[0],
        env=env,
    )
    check_junit(root / "source-distribution.xml")
    command(
        [
            python,
            "-I",
            "-m",
            "pytest",
            "-q",
            "test_installed_cli.py",
            "--junitxml=" + str(root / "installed-wheel.xml"),
        ],
        cwd=acceptance,
        env=env,
    )
    check_junit(root / "installed-wheel.xml")
    return root / "installed-wheel.xml"


def handoff(cfg, state, out):
    from oeis_learn.evaluation.discovery_gate import run_qualification, require_g9
    from oeis_learn.evaluation.learning_gate import run_qualification as learning_qualification
    from oeis_learn.evaluation.review_dispositions import require_dispositions
    from oeis_learn.evaluation.readiness import release_readiness
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

    host_capability(cfg, out)
    g9 = run_qualification(out / "discovery", native=True)
    require_g9(g9, evidence_root=out / "discovery")
    # Installed negative tests expect unrequested hardware to stay pending.
    # Keep their independently executed software G8 separate from actual G8.
    cli = out / "cli-evidence"
    shutil.copytree(state / "baseline/repairs", cli)
    learning_qualification(cli / "learning", hardware=False)
    shutil.copytree(out / "discovery", cli / "discovery")
    write(cli / "discovery-qualification-cli.json", g9)
    os.environ["OEIS_QUALIFICATION_EVIDENCE"] = str(cli)
    os.environ["OEIS_REQUIRE_NATIVE"] = "1"
    wheel = clean_wheel(out)
    command([sys.executable, ROOT / "specs/007-experiment-foundation/validate_artifacts.py"])
    paths = [
        "tests/integration/test_egca_training_step.py",
        "tests/integration/test_foundation_overflow_diagnostics.py",
        "tests/contract/test_foundation_qualified_cli.py",
    ]
    paths += [
        "tests/unit/test_foundation_" + s + ".py"
        for s in (
            "source_policy",
            "decisions",
            "legacy_wrappers",
            "measurement",
            "phase10_integration",
            "handoff",
        )
    ]
    pytest(out, paths, name="phase10")
    pytest(out, ["tests/unit/test_foundation_roles.py"], name="roles")
    dispositions = ROOT / "specs/007-experiment-foundation/validation/current-dispositions.json"
    require_dispositions(read(dispositions))
    selected = read(state / "baseline/selected.json")
    report = release_readiness(
        checkpoint=load_foundation_checkpoint(selected["checkpoint"]),
        baseline_bundle=(read(state / "baseline/bundle/baseline.json"), state / "baseline/bundle"),
        g7=(read(state / "baseline/repairs/g7.json"), state / "baseline/repairs"),
        g8=(read(state / "gpu-learning/g8.json"), state / "gpu-learning"),
        g9=(g9, out / "discovery"),
        wheel_junit=wheel,
        measurement_root=state / "measurement/run",
        release_source=(
            state / "release-source/cohort",
            archive(cfg),
            read(state / "release-source/snapshot.json"),
        ),
        phase10_junit=out / "phase10.xml",
        role_junit=out / "roles.xml",
        dispositions=dispositions,
    )
    write(out / "readiness.json", report)
    archive_results(cfg, state, out, selected, wheel, dispositions)
    return report


def archive_results(cfg, state, out, selected, wheel, dispositions):
    from oeis_learn.evaluation.handoff import archive_handoff, replay_handoff, MAX_FILES, MAX_BYTES
    from oeis_learn.evaluation.learning_gate import evidence_references
    from oeis_learn.experiments.artifacts import compute_file_hash

    selected_files = set()

    def add(file):
        file = Path(file).resolve(strict=True)
        if not file.is_file():
            raise ValueError("Handoff file required: " + str(file))
        selected_files.add(file)

    def add_tree(path, *, measurement=False):
        for p in Path(path).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if measurement and (
                p.suffix not in (".json", ".jsonl", ".pt")
                or any(
                    n in p.parts
                    for n in (
                        "runtime-queue",
                        "workers",
                        "coordination",
                        "probe-input",
                        "engineering-input",
                    )
                )
            ):
                continue
            add(p)

    for name, file in (
        ("g7", state / "baseline/repairs/g7.json"),
        ("g8", state / "gpu-learning/g8.json"),
        ("g9", out / "discovery/g9.json"),
    ):
        add(file)
        report = read(file)
        refs = evidence_references(report) if name == "g8" else report["checks"].values()
        for ref in refs:
            add(file.parent / ref["path"])
    add_tree(state / "baseline/bundle")
    add_tree(Path(selected["checkpoint"]).parent)
    add_tree(state / "measurement/run", measurement=True)
    add_tree(state / "release-source/cohort")
    add_tree(cfg["archive"])
    for p in (
        wheel,
        out / "phase10.xml",
        out / "roles.xml",
        dispositions,
        cfg["holdouts"],
        state / "release-source/snapshot.json",
    ):
        add(p)
    root = Path(os.path.commonpath([str(p.parent) for p in selected_files]))
    if not root.is_relative_to("/workspace"):
        raise ValueError("Handoff inputs must share an accepted /workspace root")
    if len(selected_files) > MAX_FILES or sum(p.stat().st_size for p in selected_files) > MAX_BYTES:
        raise Blocked("Actual handoff exceeds unchanged 4096-file/40-GiB bound")

    def relative(path):
        return Path(path).resolve().relative_to(root).as_posix()

    readiness = dict(
        checkpoint=relative(selected["checkpoint"]),
        baseline_bundle=relative(state / "baseline/bundle/baseline.json"),
        g7=relative(state / "baseline/repairs/g7.json"),
        g8=relative(state / "gpu-learning/g8.json"),
        g9=relative(out / "discovery/g9.json"),
        wheel_junit=relative(wheel),
        phase10_junit=relative(out / "phase10.xml"),
        role_junit=relative(out / "roles.xml"),
        dispositions=relative(dispositions),
        measurement_root=relative(state / "measurement/run"),
        release_source=dict(
            cohort=relative(state / "release-source/cohort"),
            archive=relative(cfg["archive"]),
            holdouts=relative(cfg["holdouts"]),
            snapshot=relative(state / "release-source/snapshot.json"),
        ),
    )
    manifest = dict(
        version="handoff-inputs/v1",
        files=[dict(path=relative(p), sha256=compute_file_hash(p)) for p in sorted(selected_files)],
        readiness=readiness,
    )
    write(out / "handoff-inputs.json", manifest)
    report = archive_handoff(manifest, input_root=root, output=out / "handoff")
    if replay_handoff(out / "handoff") != report:
        raise ValueError("Immutable handoff replay changed")


def minimal_cohort(release, out):
    """Keep all members of <=2 original groups per split, chosen before training."""
    from oeis_learn.evaluation.foundation_cohort import load_cohort, freeze_cohort
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    parent = load_cohort(release / "cohort")
    cohort = out / "cohort"
    if (cohort / "manifest.json").exists():
        load_cohort(cohort)
        return cohort
    groups = [
        g
        for partition in ("development", "final")
        for g in [r for r in parent["groups"] if r["partition"] == partition][:2]
    ]
    if {g["partition"] for g in groups} != {"development", "final"}:
        raise Blocked("Both original development and final groups required")
    members = {m for g in groups for m in g["members"]}
    rows = [r for r in read(release / "source/records.json") if r["record_id"] in members]
    stage = out / "cohort-staging"
    if stage.exists():
        shutil.rmtree(stage)
    write(stage / "source/records.json", rows)
    config = dict(
        parent["config"],
        dev_count=sum(g["partition"] == "development" for g in groups),
        final_count=sum(g["partition"] == "final" for g in groups),
    )
    write(stage / "config.json", config)
    freeze_cohort(stage / "source", stage / "config.json", stage / "cohort")
    actual = load_cohort(stage / "cohort")
    if {(g["group_id"], g["partition"]) for g in actual["groups"]} != {
        (g["group_id"], g["partition"]) for g in groups
    }:
        raise ValueError("Diagnostic projection changed group identity/partition")
    write(
        stage / "cohort/projection.json",
        dict(
            parent_cohort_id=parent["cohort_id"],
            selected_groups=[g["group_id"] for g in groups],
            selection_sha256=digest(config),
        ),
    )
    (stage / "cohort").rename(cohort)
    return cohort


def end_to_end(cfg, state, out):
    from oeis_learn.evaluation.coordinated_smoke import coordinated_smoke
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation, default_protocol
    from oeis_learn.evaluation.foundation_cohort import load_cohort
    from oeis_learn.evaluation.finalization import create_finalization, final_output
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
    from oeis_learn.tracking.foundation_metrics import EventLog
    import statistics

    host_capability(cfg, out)
    cohort = minimal_cohort(state / "release-source", out)
    protocol = out / "protocol.json"
    if not protocol.exists():
        write(protocol, dict(default_protocol(), attempts=2, max_body_tokens=128))

    def segment(stop_after):
        previous = sorted((out / "segments").glob("*/smoke/run"))
        resume = previous[-1] if previous else None
        if resume is not None and not list((resume / "checkpoints").glob("checkpoint-[0-9]*.json")):
            raise Blocked(
                "Interrupted initial preparation has no recoverable checkpoint; preserve its budget evidence"
            )
        destination = out / "segments" / str(time.time_ns())
        report = coordinated_smoke(
            ROOT / "configs/foundation/wat_smoke.yaml",
            state / "baseline/pool",
            cfg["image_id"],
            destination,
            resume_run=resume,
            stop_after=stop_after,
            run_provenance=provenance(cfg, state),
        )
        run = destination / "smoke/run"
        return dict(
            report["result"],
            checkpoint=str(run / "checkpoints" / report["result"]["checkpoint_name"]),
            run=str(run),
        )

    if not (out / "pause.json").exists():
        write(out / "pause.json", segment(1))
    if not (out / "trained.json").exists():
        trained = segment(None)
        if trained["completed_updates"] != 3:
            raise ValueError("Three frozen updates required")
        write(out / "trained.json", trained)
    result = read(out / "trained.json")
    run = Path(result["run"])
    checkpoint = load_foundation_checkpoint(
        result["checkpoint"], device="cpu", include_payload=True
    )
    if not (out / "development.json").exists():
        write(
            out / "development.json",
            evaluate_foundation(
                result["checkpoint"],
                cohort,
                protocol,
                out / "development",
                split="development",
                device="cuda",
            ),
        )
    if not (out / "stopping.json").exists():
        write(
            out / "stopping.json",
            dict(
                decision="stop",
                reason="Frozen minimal three-update GPU lifecycle completed; no selection on final truth",
                checkpoint_sha256=checkpoint.manifest["blob_sha256"],
                protocol_sha256=digest(read(protocol)),
                cohort_id=load_cohort(cohort)["cohort_id"],
            ),
        )
    finalization = create_finalization(
        run,
        result["checkpoint"],
        protocol,
        cohort,
        out / "stopping.json",
        out / "finalization.json",
    )
    final = evaluate_foundation(
        result["checkpoint"],
        cohort,
        protocol,
        final_output(finalization),
        split="final",
        finalization=out / "finalization.json",
        device="cuda",
    )
    updates = [r for r in EventLog(run / "events").events.values() if r.get("kind") == "update"]
    if len(updates) != 3 or not all(
        r["parameters_changed"] and r["parameter_device"].startswith("cuda") for r in updates
    ):
        raise ValueError("Three actual changed-parameter HIP updates required")
    times = [sum(r["phase_timings_ns"].values()) / 1e9 for r in updates]
    metrics = dict(
        timed_update_seconds=times,
        median_timed_update_seconds=statistics.median(times),
        nonpadding_tokens=sum(r["tokens"]["value"] for r in updates),
        measured_phase_tokens_per_second=sum(r["tokens"]["value"] for r in updates) / sum(times),
        parameter_count=sum(
            p.numel() for m in (checkpoint.encoder, checkpoint.decoder) for p in m.parameters()
        ),
        checkpoint_bytes=Path(result["checkpoint"]).with_suffix(".pt").stat().st_size,
        extrapolation="Not a steady-state estimate; compare T119 ten-minute measurements before extending budgets",
    )
    report = dict(
        purpose="minimal_gpu_lifecycle_diagnostic",
        qualified=False,
        completed_updates=3,
        planning_metrics=metrics,
        pause=read(out / "pause.json"),
        resume=result,
        checkpoint_sha256=checkpoint.manifest["blob_sha256"],
        updates=updates,
        development=read(out / "development.json"),
        final=final,
        training_ledger=read(run / "ledger.json"),
        lifetime_budget=read(state / "final-budget.json"),
    )
    write(out / "planning-report.json", report)
    return report


def execute(name, cfg, state, out, seconds):
    if name in HOST_STEPS:
        raise Blocked(ISOLATED_ROLES_BLOCKER)
    if name == "inventory":
        return inventory(out)
    if name == "cpu-regressions":
        return cpu_regressions(out)
    if name == "image":
        return ensure_image(cfg)
    if name == "phase13-inputs":
        verify = runpy.run_path(str(ROOT / "scripts/verify_phase13.py"))["verify_inputs"]
        return verify(
            out,
            count=256,
            archive=archive(cfg) if cfg.get("archive") and cfg.get("holdouts") else None,
        )
    if name == "release-source":
        from oeis_learn.evaluation.release_source import prepare_release, audit_release

        prepare_release(cfg["source_manifest"], cfg["cohort_config"], out)
        a = archive(cfg)
        snapshot = a.freeze()
        report = audit_release(out / "cohort", archive=a, snapshot=snapshot)
        write(out / "snapshot.json", snapshot)
        write(out / "release-source-audit.json", report)
        if report["status"] != "PASS":
            raise Blocked("Authoritative source/structural archive did not qualify")
        return report
    if name == "gpu-environment":
        return gpu_environment(out, cfg)
    if name == "gpu-smoke":
        # The broker's /tmp is a 2 GiB tmpfs. Keep the full training run on
        # workspace storage so its frozen 100 GiB free-space guard measures the
        # actual run filesystem, and retain its diagnostics in the stage receipt.
        env = dict(os.environ, OEIS_GPU_SMOKE_RUN_DIR=str((out / "training-run").resolve()))
        return pytest(
            out,
            [
                "tests/integration/test_foundation_gpu.py::test_actual_hip_three_updates",
                "tests/integration/test_foundation_evaluation.py::test_gpu_checkpoint_loading_and_prefix_only_generation",
                "tests/unit/test_foundation_sampler.py::test_complete_effective_constructors_and_real_forward_backward[cuda]",
            ],
            name="gpu-smoke",
            env=env,
        )
    if name == "gpu-learning":
        from oeis_learn.evaluation.learning_gate import run_qualification, require_g8

        report = run_qualification(out, hardware=True)
        require_g8(report, evidence_root=out)
        return report
    if name == "host-containment":
        host_capability(cfg, out)
        return pytest(
            out,
            [
                "tests/unit/test_foundation_roles.py",
                "tests/integration/test_foundation_resources.py",
            ],
            name="containment",
        )
    if name == "preflight":
        from oeis_learn.cli.foundation_preflight import main

        config = preflight_config(cfg, out)
        write(out / "config-reference.json", dict(path=str(config)))
        code = main(["--config", str(config), "--output", str(out), "--hardware-only"])
        if code:
            raise RuntimeError(f"Actual preflight exit {code}; inspect preflight-report.json")
        report = read(out / "preflight-report.json")
        if report["status"] != "ready":
            raise ValueError("Actual ready preflight required")
        return report
    if name == "baseline":
        return baseline(cfg, state, out)
    if name in ("legacy-cpu", "legacy-gpu"):
        return legacy(cfg, out, "cuda" if name == "legacy-gpu" else "cpu")
    if name == "overflow-gpu":
        return overflow(cfg, state, out)
    if name == "measurement":
        return measurement(cfg, state, out)
    if name == "release-handoff":
        return handoff(cfg, state, out)
    if name == "end-to-end":
        return end_to_end(cfg, state, out)
    raise ValueError("Unknown fixed worker step: " + name)
