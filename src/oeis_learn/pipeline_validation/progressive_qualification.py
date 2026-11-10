"""Serial actual-HIP qualification of exact selected model/input/update scopes."""

import os
from pathlib import Path
import time
from oeis_learn.generalization.progressive.contracts import digest, seal, atomic_write, load


def matrix(
    profiles, *, source, image, engine, input_lengths=(4, 20), microbatches=(7, 16), hardware=None
):
    from oeis_learn.generalization.progressive.models import validate_profile

    import re

    if (
        not isinstance(source, str)
        or not isinstance(image, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", source)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
    ):
        raise ValueError("exact image/source digests required")
    if (
        engine not in {"docker", "podman"}
        or not image.startswith("sha256:")
        or not source.startswith("sha256:")
    ):
        raise ValueError("identified source/image/engine required")
    if (
        not profiles
        or not input_lengths
        or not microbatches
        or len(profiles) > 12
        or any(type(n) is not int or n not in (4, 20) for n in input_lengths)
        or any(type(n) is not int or n not in (7, 16) for n in microbatches)
    ):
        raise ValueError("bounded selected scope matrix required")
    cases = {}
    for profile in profiles:
        validate_profile(profile)
        for length in input_lengths:
            for microbatch in microbatches:
                case = seal(
                    dict(
                        profile=profile,
                        input_length=length,
                        microbatch=microbatch,
                        precision="float32",
                        update="canonical-logical-update/v1",
                    )
                )
                cases[case["identity"]] = case
    if len(cases) > 48:
        raise ValueError("qualification matrix budget")
    return seal(
        dict(
            version="progressive-hip-matrix/v1",
            source=source,
            image=image,
            engine=engine,
            serial=True,
            hardware=hardware,
            cases=[cases[k] for k in sorted(cases)],
        )
    )


def require_hip():
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("actual HIP build/runtime required; CPU fallback forbidden")
    device = torch.empty(1, device="cuda")
    if device.device.type != "cuda":
        raise RuntimeError("device tensor required")
    return torch


def _same(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _cleanup_memory(torch):
    """Release owned garbage and the documented HIP BLAS workspace cache.

    empty_cache alone cannot release hipBLAS workspaces. Keep the allocation
    counts on both sides so workspace retention cannot conceal live tensors.
    https://docs.pytorch.org/docs/2.9/notes/hip.html#hipblas-workspaces
    """
    import gc

    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return dict(
        before_workspace_release_bytes=before,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        allocator_stats=torch.cuda.memory_stats(),
    )


def run_case(request, case, output, *, client):
    """One original producer; missing devices and incompatible reuse fail closed."""
    torch = require_hip()
    from oeis_learn.experiments.source_snapshot import source_snapshot

    actual_source = digest(source_snapshot())
    expected = matrix(
        list({c["profile"]["identity"]: c["profile"] for c in request["cases"]}.values()),
        source=request["source"],
        image=request["image"],
        engine=request["engine"],
        input_lengths=tuple(sorted({c["input_length"] for c in request["cases"]})),
        microbatches=tuple(sorted({c["microbatch"] for c in request["cases"]})),
        hardware=request["hardware"],
    )
    if (
        seal(request) != request
        or case not in request["cases"]
        or request != expected
        or request["source"] != actual_source
    ):
        raise ValueError("current exact qualification matrix required")
    if request["hardware"] != hardware_facts():
        raise ValueError("exact current HIP hardware/runtime facts required")
    if os.environ.get("OEIS_QUALIFICATION_IMAGE") != request["image"]:
        raise ValueError("original running image identity required")
    if client is None:
        raise ValueError("actual packaged native worker required")
    scope = digest(
        [request["source"], request["image"], request["engine"], request["hardware"], case]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    producer = output / "producer.json"
    if producer.exists():
        value = load(producer)
        if (
            seal(value) != value
            or value.get("source") != actual_source
            or value.get("scope") != scope
            or value.get("status") != "PASS"
        ):
            raise ValueError("incompatible original qualification producer")
        return value
    from oeis_learn.pipeline_validation.progressive_training import (
        PolicyUpdate,
        resolve_update_policy,
        RecoverableUpdate,
    )
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.decoder.bigint_codec import BOS_ID, PAD_ID, encode_body
    from oeis_learn.pipeline_validation.progressive_evaluation import generate_short, score_short
    import numpy as np
    import random

    began = time.monotonic()
    baseline_allocated = _cleanup_memory(torch)["allocated_bytes"]
    torch.cuda.reset_peak_memory_stats()
    policy = resolve_update_policy(
        case["profile"], schedule_horizon_tokens=10000, microbatch_size=case["microbatch"]
    )
    runner = PolicyUpdate(case["profile"], policy, device="cuda")
    source = "(program (return (add $index (int 1))))"
    sequence = list(range(1, case["input_length"] + 1))
    sequences = [sequence] * 16
    ids = [BOS_ID] + encode_body(source)
    tokens = torch.tensor([ids] * 16, device="cuda")
    charge = int(tokens[:, 1:].ne(PAD_ID).sum())
    before = [
        v.detach().clone() for v in [*runner.encoder.parameters(), *runner.decoder.parameters()]
    ]
    _, ledger = runner.update(sequences, tokens, ledger=Ledger(10000).reserve("first", charge))
    changed = any(
        not torch.equal(a, b)
        for a, b in zip(before, [*runner.encoder.parameters(), *runner.decoder.parameters()])
    )
    states = dict(
        weights=dict(encoder=runner.encoder.state_dict(), decoder=runner.decoder.state_dict()),
        optimizer=runner.state_dict(),
        scheduler=dict(cursor=1),
        data=dict(cursor=16, pending_batch=None),
        replay=dict(cursor=0),
        history={},
        pins={},
        selection=dict(scope=scope),
        rng=dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            device=torch.cuda.get_rng_state_all(),
        ),
    )
    store = GeneralizationCheckpointStore(output / "checkpoints", scope, 10000)
    record = store.save(states, ledger)
    restored = store.load(record)
    runner.update(sequences, tokens, ledger=ledger.reserve("second", charge))
    gold = [
        v.detach().clone() for v in [*runner.encoder.parameters(), *runner.decoder.parameters()]
    ]
    gold_optimizer = runner.state_dict()
    resumed = PolicyUpdate(case["profile"], policy, device="cuda")
    resumed.encoder.load_state_dict(restored["states"]["weights"]["encoder"])
    resumed.decoder.load_state_dict(restored["states"]["weights"]["decoder"])
    resumed.load_state_dict(restored["states"]["optimizer"])
    RecoverableUpdate._rng_restore(restored["states"]["rng"])
    resumed.update(sequences, tokens, ledger=restored["ledger"].reserve("second", charge))
    deterministic = all(
        torch.equal(a, b)
        for a, b in zip(gold, [*resumed.encoder.parameters(), *resumed.decoder.parameters()])
    ) and _same(gold_optimizer, resumed.state_dict())
    if not changed or not deterministic:
        raise ValueError("HIP update/restore deterministic continuation failed")
    memory = None
    if case["input_length"] == 4:
        generated = generate_short(
            resumed.encoder,
            resumed.decoder,
            [dict(indices=["0", "1", "2", "3"], values=["1", "2", "3", "4"])],
            seed=13,
            max_tokens=32,
        )
        scored = score_short(
            source,
            dict(indices=list(map(str, range(9))), values=list(map(str, range(1, 10)))),
            client=client,
        )
        if not scored["agreement"]:
            raise ValueError("short native score failed")
    else:
        from oeis_learn.decoder.bigint_generation import GenerationRequest, sample_batch

        resumed.encoder.eval()
        resumed.decoder.eval()
        with torch.no_grad():
            memory = resumed.encoder.forward_from_sequences([sequence], device="cuda")
        generated = sample_batch(
            resumed.decoder, [GenerationRequest(memory, 13, 0.8, time.monotonic() + 4, 32)]
        )
        checked = client.execute(source, list(map(str, range(100))), deadline=time.monotonic() + 4)
        if checked.get("values") != list(map(str, range(1, 101))):
            raise ValueError("native qualification score failed")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    parameters = sum(
        v.numel() for v in [*resumed.encoder.parameters(), *resumed.decoder.parameters()]
    )
    generation = bool(generated and generated[0]["tokens"])
    # Avoid locals(): on Python 3.12 its snapshot would retain memory even
    # after `del memory`, turning the cleanup probe itself into an owner.
    del runner, resumed, before, gold, gold_optimizer, states, restored, tokens, memory
    cleanup = _cleanup_memory(torch)
    after_cleanup = cleanup["allocated_bytes"]
    measures = dict(
        finite_update=True,
        parameter_change=changed,
        full_restore=deterministic,
        deterministic_continuation=deterministic,
        generation=generation,
        native_reference_score=True,
        cleanup=after_cleanup <= baseline_allocated,
        parameters=parameters,
        peak_allocated_bytes=peak,
        cleanup_allocated_bytes=after_cleanup,
        baseline_allocated_bytes=baseline_allocated,
        cleanup_diagnostics=cleanup,
        elapsed_us=int((time.monotonic() - began) * 1e6),
    )
    # Retain successful operations and exact allocator counters even on failure.
    atomic_write(output / "measurements.json", measures)
    if not measures["cleanup"]:
        raise ValueError(
            f"owned HIP tensors remain after cleanup: {after_cleanup} bytes "
            f"versus baseline {baseline_allocated}; see measurements.json"
        )
    if not generation:
        raise ValueError("actual checkpoint generation absent")
    value = seal(
        dict(
            version="progressive-hip-case/v1",
            source=actual_source,
            scope=scope,
            device="actual_HIP",
            status="PASS",
            requested=[
                "finite_update",
                "parameter_change",
                "full_restore",
                "deterministic_continuation",
                "generation",
                "native_reference_score",
                "cleanup",
            ],
            executed=[
                "finite_update",
                "parameter_change",
                "full_restore",
                "deterministic_continuation",
                "generation",
                "native_reference_score",
                "cleanup",
            ],
            # Keep the closed v1 authority receipt unchanged. Allocator details
            # belong in measurements.json, retained on both success and failure.
            measurements={k: v for k, v in measures.items() if k != "cleanup_diagnostics"},
        )
    )
    atomic_write(producer, value)
    return value


def qualify(request, output, *, client):
    """Serial GPU owner; compatible original completed cases can be reused."""
    require_hip()
    results = []
    for case in request["cases"]:
        results.append(
            run_case(request, case, Path(output) / case["identity"].split(":")[1], client=client)
        )
    result = seal(
        dict(
            version="progressive-hip-qualification/v1",
            matrix=request["identity"],
            requested=[c["identity"] for c in request["cases"]],
            executed=[c["identity"] for c in request["cases"]],
            producers=[r["identity"] for r in results],
            status="PASS",
            device="actual_HIP",
        )
    )
    summary = Path(output) / "qualification.json"
    if summary.exists():
        if load(summary) != result:
            raise ValueError("changed original qualification inventory")
    else:
        atomic_write(summary, result)
    return result


def hardware_facts():
    torch = require_hip()
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        driver = Path("/sys/module/amdgpu/version").read_text().strip()
    except OSError:
        driver = None
    return seal(
        dict(
            version="progressive-hip-hardware/v1",
            name=p.name,
            architecture=getattr(p, "gcnArchName", None),
            uuid=str(getattr(p, "uuid", "unavailable")),
            total_memory=p.total_memory,
            torch=str(torch.__version__),
            hip=str(torch.version.hip),
            driver=driver,
            driver_unavailable_reason=None if driver else "amdgpu module version not exposed",
        )
    )


def calibrate_iteration(package,output,*,source,image,device='cuda',checkpoint_bytes=1 << 30,seconds=600):
    """Measure exact prepared logical-update/probe/publication scope before freeze.

    The caller's qualification supervisor owns the deadline. CPU receipts remain
    diagnostic and cannot satisfy the research iteration-margin validator.
    """
    if (type(seconds) not in (int, float) or not 0 < seconds <= 3600
            or type(checkpoint_bytes) is not int or not 0 < checkpoint_bytes <= 64 << 30):
        raise ValueError('finite calibration time/checkpoint allowance required')
    deadline = time.monotonic() + seconds
    def within_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('calibration deadline exhausted')
    import torch
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate
    from oeis_learn.pipeline_validation.generalization_curriculum import observe_targets
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.decoder.bigint_codec import BOS_ID,PAD_ID
    if device=='cuda':
        require_hip()
        from oeis_learn.experiments.source_snapshot import source_snapshot
        if source!=digest(source_snapshot()) or os.environ.get('OEIS_QUALIFICATION_IMAGE')!=image:raise ValueError('current original calibration source/image required')
    elif device!='cpu':raise ValueError('explicit qualification device required')
    if seal(package)!=package:raise ValueError('original prepared calibration input required')
    output=Path(output)
    if output.exists():raise ValueError('fresh iteration calibration output required')
    output.mkdir(parents=True)
    rows=[r for shard in package['shards'] for r in shard]
    longest=max(rows,key=lambda r:len(r['tokens']))
    sequence=[int(max((r['values'][i] for r in rows),key=lambda v:(abs(int(v)).bit_length(),abs(int(v))))) for i in range(4 if package.get('task_profile')=='combinatorial-short/v1' else 20)]
    count=package['update_policy']['effective_batch']
    ids=[BOS_ID,*longest['tokens']]
    tokens=torch.tensor([ids]*count,device=device)
    charge=int(tokens[:,1:].ne(PAD_ID).sum())
    within_deadline()
    runner=PolicyUpdate(package['profile'],package['update_policy'],device=device)
    measurements=dict(update_us=0,probe_us=0,checkpoint_us=0)
    def now():
        if device=='cuda':torch.cuda.synchronize()
        return time.monotonic_ns()//1000
    ledger=Ledger(3*charge)
    store=GeneralizationCheckpointStore(output/'checkpoints',compatibility(package),3*charge)
    store.MAX_BYTES = 2 * checkpoint_bytes
    for iteration in range(2):
        within_deadline()
        start=now()
        _,ledger=runner.update([sequence]*count,tokens,ledger=ledger.reserve(str(iteration),charge))
        measurements['update_us']=max(measurements['update_us'],now()-start)
        start=now()
        # An allocation boundary can perform both the old and new block probe.
        for _ in range(2):observe_targets(runner.encoder,runner.decoder,[dict(longest,values=[str(v) for v in sequence])],device=device)
        measurements['probe_us']=max(measurements['probe_us'],now()-start)
        start=now()
        states=dict(weights=dict(encoder=runner.encoder.state_dict(),decoder=runner.decoder.state_dict()),optimizer=runner.state_dict(),scheduler={},data=dict(cursor=ledger.updates,pending_batch=None),replay={},history={},pins={},selection={},rng=dict(python=__import__('random').getstate(),numpy=__import__('numpy').random.get_state(),torch=torch.get_rng_state(),device=torch.cuda.get_rng_state_all() if device=='cuda' else []))
        # Bound the complete serialized generation before the first tensor write.
        # The additional weights allowance matches checkpoint_size's conservative
        # optimizer-growth headroom, even after Adam has allocated its moments.
        pieces = [v for k,v in states.items() if k not in {'rng','data'}] + list(states['rng'].values())
        measured = sum(len(store._encode(v)) for v in pieces) + 2*len(store._encode(states['weights'])) + (128 << 10)
        if measured > checkpoint_bytes:
            raise TimeoutError('calibration checkpoint exceeds prospective storage allowance')
        from oeis_learn.generalization.progressive.storage import preflight
        preflight(measured, len(pieces)+4)
        within_deadline()
        ref=store.save(states,ledger);store.load(ref)
        measurements['checkpoint_us']=max(measurements['checkpoint_us'],now()-start)
    from oeis_learn.generalization.progressive.storage import checkpoint_size
    checkpoint_bytes,checkpoint_inodes=checkpoint_size(output/'checkpoints',ref,checkpoint_root='.',research=True)
    result=seal(dict(version='progressive-iteration-calibration/v1',checkpoint_bytes=checkpoint_bytes,checkpoint_inodes=checkpoint_inodes,source=source,image=image,device='actual_HIP' if device=='cuda' else 'cpu',status='PASS',compatibility=compatibility(package),maximum_target_tokens=len(longest['tokens']),effective_batch=count,measurements={k:max(1,v) for k,v in measurements.items()},measurement_scope='two complete logical updates including initial Adam allocation; two probes; durable full-state save and verified restore; maximal prepared target and numeric input sizes'))
    within_deadline()
    atomic_write(output/'producer.json',result)
    # Release owned tensors between serial prepared-input cases.
    del runner, states, tokens, pieces
    if device == 'cuda':
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result
