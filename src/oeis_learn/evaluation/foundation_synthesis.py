"""Checkpoint-driven, prefix-only proposal processes and sealed finite scoring.

No legacy canary, teacher, solver or metadata-scaffold API is imported here.
Docker mount enforcement is integrated by T046; the process boundary here uses
spawn and sends only a validated VisiblePrompt and public sampling parameters.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import multiprocessing as mp
from pathlib import Path
import socket
import time

from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    canonical_bytes,
    compute_canonical_digest as digest,
    compute_file_hash,
    load_json,
)
from oeis_learn.experiments.models import validate_artifact, parse_integer_text
from oeis_learn.experiments.profiles import profile_digests, I256_MIN, I256_MAX
from oeis_learn.evaluation.foundation_cohort import exact_keys, write_json, read_ref, load_cohort
from oeis_learn.sandbox.pipeline import Limits, verify, runtime_identity
from oeis_learn.sandbox.runner import FoundationRunner
from oeis_learn.sandbox.worker_pool import _send, _receive


class EvaluationGateError(ValueError):
    """A persisted decision/evidence boundary failed; no score may qualify."""


class HardwareUnavailable(RuntimeError):
    """The explicitly requested model device is unavailable."""


def default_protocol():
    return {
        "schema_version": "foundation/v1",
        "kind": "evaluation_protocol",
        "run_seed": 20260913,
        "language_profile": profile_digests()["language"],
        "codec_profile": profile_digests()["codec"],
        "attempts": 16,
        "temperature": 0.8,
        "top_p": 1.0,
        "max_body_tokens": 1024,
        "prefix_budget_ns": 20_000_000_000,
        "full_budget_ns": 10_000_000_000,
        "selector": "body_token_count,source_sha256",
    }


def validate_protocol(protocol):
    required = set(default_protocol())
    if (
        not isinstance(protocol, dict)
        or not required <= protocol.keys()
        or set(protocol) - required - {"checkpoint_sha256", "cohort_sha256"}
    ):
        raise ValueError("missing/unknown evaluation protocol field")
    if type(protocol["top_p"]) not in (int, float):
        raise ValueError("top_p must be numeric, not boolean")
    baseline = default_protocol()
    for key in ("schema_version", "kind", "language_profile", "codec_profile", "top_p", "selector"):
        if protocol[key] != baseline[key]:
            raise ValueError(f"incompatible protocol {key}")
    for key, cap in [
        ("attempts", 16),
        ("max_body_tokens", 1024),
        ("prefix_budget_ns", 20_000_000_000),
        ("full_budget_ns", 10_000_000_000),
    ]:
        if type(protocol[key]) is not int or not 1 <= protocol[key] <= cap:
            raise ValueError(f"invalid protocol {key}")
    if (
        type(protocol["run_seed"]) is not int
        or protocol["run_seed"] < 0
        or type(protocol["temperature"]) not in (int, float)
        or protocol["temperature"] != 0.8
    ):
        raise ValueError("invalid seed or temperature")
    return protocol


def sampling_projection(protocol):
    # Deliberately enumerate rather than subtract private fields.
    return {
        key: protocol[key]
        for key in (
            "language_profile",
            "codec_profile",
            "attempts",
            "temperature",
            "top_p",
            "max_body_tokens",
            "prefix_budget_ns",
            "full_budget_ns",
            "selector",
        )
    }


def attempt_seed(protocol, visible_terms, attempt_index):
    prompt_hash = digest(visible_terms)
    data = [protocol["run_seed"], digest(sampling_projection(protocol)), prompt_hash, attempt_index]
    return int.from_bytes(hashlib.sha256(canonical_bytes(data)).digest()[:8], "big")


def _model_worker(channel, checkpoint_path, device):
    import torch
    from oeis_learn.decoder.sampler import WatProgramSampler
    from oeis_learn.decoder.program_codec import CodecError
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

    torch.set_num_threads(4)
    try:
        checkpoint = load_foundation_checkpoint(checkpoint_path, device=device)
        sampler = WatProgramSampler(
            checkpoint.decoder, codec_profile="wat_body_decimal_v1", top_p=1.0
        )
        _send(channel, {"type": "ready", "checkpoint_sha256": checkpoint.manifest["blob_sha256"]})
        while True:
            request = _receive(channel)
            exact_keys(
                request, ("prompt", "seed", "temperature", "max_body_tokens"), "generation request"
            )
            prompt = request["prompt"]
            validate_artifact(prompt)
            if prompt["kind"] != "visible_prompt":
                raise ValueError("only VisiblePrompt allowed")
            try:
                with torch.no_grad():
                    memory = checkpoint.encoder.forward_from_sequences(
                        [[int(x) for x in prompt["observed_terms"]]], device=checkpoint.device
                    )
                    if isinstance(memory, tuple):
                        memory = memory[0]
                    source, tokens = sampler.sample_candidate(
                        memory,
                        seed=request["seed"],
                        temperature=request["temperature"],
                        top_p=1.0,
                        max_length=request["max_body_tokens"],
                    )
                _send(
                    channel,
                    {
                        "source": source,
                        "tokens": tokens.cpu().tolist(),
                        "outcome": None,
                        "reason": None,
                    },
                )
            except CodecError as exc:
                _send(
                    channel,
                    {
                        "source": None,
                        "tokens": [],
                        "outcome": "execution_limit" if "cap" in str(exc) else "invalid_syntax",
                        "reason": str(exc),
                    },
                )
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        _send(channel, {"type": "error", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        channel.close()


class ModelProcess:
    """Persistent actual model, with an external per-attempt remaining deadline."""

    def __init__(self, checkpoint_path, *, device="cpu"):
        self.path, self.device = str(Path(checkpoint_path).resolve()), device
        self.process = self.channel = None
        self.start()

    def start(self, deadline_ns=None):
        parent, child = socket.socketpair()
        self.process = mp.get_context("spawn").Process(
            target=_model_worker, args=(child, self.path, self.device), daemon=True
        )
        self.process.start()
        child.close()
        self.channel = parent
        try:
            ready = _receive(
                parent,
                min(time.monotonic_ns() + 60_000_000_000, deadline_ns)
                if deadline_ns is not None
                else time.monotonic_ns() + 60_000_000_000,
            )
            if ready.get("type") != "ready":
                raise ValueError(ready.get("reason", "model startup failed"))
            self.checkpoint_sha256 = ready["checkpoint_sha256"]
        except BaseException:
            self.close()
            raise

    def generate(self, prompt, *, seed, temperature, max_body_tokens, deadline_ns):
        validate_artifact(prompt)
        try:
            if self.process is None:
                self.start(deadline_ns)
            self.channel.settimeout(max(1e-9, (deadline_ns - time.monotonic_ns()) / 1e9))
            _send(
                self.channel,
                {
                    "prompt": prompt,
                    "seed": seed,
                    "temperature": temperature,
                    "max_body_tokens": max_body_tokens,
                },
            )
            result = _receive(self.channel, deadline_ns)
            if result.get("type") == "error":
                raise RuntimeError(result["reason"])
            exact_keys(result, ("source", "tokens", "outcome", "reason"), "model result")
            return result
        except TimeoutError:
            self.close()
            return {
                "source": None,
                "tokens": [],
                "outcome": "execution_limit",
                "reason": "generation_deadline",
            }
        except (OSError, EOFError) as exc:
            self.close()
            raise RuntimeError("generation infrastructure failure") from exc

    def close(self):
        if self.channel is not None:
            self.channel.close()
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
            self.process.join(2)
            if self.process.is_alive():
                raise RuntimeError("model process reclamation exceeded two seconds")
            self.process.close()
        self.process = self.channel = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _remaining(root, name, budget):
    """Conservatively charge downtime after interruption; never refund a phase."""
    path = ArtifactPath(root, name).as_path()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not path.exists():
        write_json(
            root,
            name,
            {
                "started_utc_ns": time.time_ns(),
                "started_monotonic_ns": time.monotonic_ns(),
                "boot_id": boot_id,
                "budget_ns": budget,
            },
        )
    record = load_json(path.read_bytes())
    exact_keys(
        record, ("started_utc_ns", "started_monotonic_ns", "boot_id", "budget_ns"), "phase timer"
    )
    if record["budget_ns"] != budget:
        raise ValueError("phase budget changed on resume")
    elapsed = (
        time.monotonic_ns() - record["started_monotonic_ns"]
        if record["boot_id"] == boot_id
        else time.time_ns() - record["started_utc_ns"]
    )
    if elapsed < 0:
        raise ValueError("clock moved backwards; cannot refund budget")
    return max(0, budget - elapsed)


def _load_seal(path, identity, protocol, prompt, group_id, checkpoint_sha):
    from oeis_learn.decoder.program_codec import encode_body

    seal = load_json(path.read_bytes())
    exact_keys(
        seal,
        (
            "schema_version",
            "evaluation_id",
            "group_id",
            "prompt_sha256",
            "checkpoint_sha256",
            "finalization_id",
            "attempts",
            "selected",
            "generation_complete",
            "selector",
            "seal_id",
        ),
        "candidate seal",
    )
    if (
        seal["schema_version"] != "foundation/v1"
        or seal["evaluation_id"] != identity
        or seal["seal_id"] != digest(seal, "seal_id")
        or seal["group_id"] != group_id
        or seal["checkpoint_sha256"] != checkpoint_sha
        or seal["prompt_sha256"] != digest(prompt["observed_terms"])
        or seal["selector"] != protocol["selector"]
        or type(seal["generation_complete"]) is not bool
    ):
        raise EvaluationGateError("candidate seal identity mismatch")
    attempts = seal["attempts"]
    if not isinstance(attempts, list) or len(attempts) > protocol["attempts"]:
        raise EvaluationGateError("invalid sealed attempt count")
    for index, attempt in enumerate(attempts):
        exact_keys(
            attempt,
            (
                "attempt_index",
                "seed",
                "source",
                "tokens",
                "outcome",
                "reason",
                "source_sha256",
                "prefix_result",
                "generation_elapsed_ns",
                "generation_peak_memory",
            ),
            "sealed attempt",
        )
        if (
            type(attempt["attempt_index"]) is not int
            or attempt["attempt_index"] != index
            or attempt["seed"] != attempt_seed(protocol, prompt["observed_terms"], index)
        ):
            raise EvaluationGateError("invalid sealed attempt order/seed")
        if attempt["source"] is not None and encode_body(attempt["source"]) != attempt["tokens"]:
            raise EvaluationGateError("sealed source/token mismatch")
        result = attempt["prefix_result"]
        if result is not None:
            validate_artifact(result)
            if (
                result["purpose"] != "model"
                or result["checkpoint_sha256"] != checkpoint_sha
                or result["attempt_index"] != index
                or result["prompt_sha256"] != seal["prompt_sha256"]
                or result["source_sha256"]
                != "sha256:" + hashlib.sha256(attempt["source"].encode()).hexdigest()
                or result["outcome"] != attempt["outcome"]
                or result["selected"] != (index == seal["selected"])
                or result["stage"] not in ("prefix", "prepare")
                or result["expected_terms_sha256"] != digest({"terms": prompt["observed_terms"]})
            ):
                raise EvaluationGateError("sealed prefix evidence mismatch")
        elif attempt["outcome"] not in ("invalid_syntax", "execution_limit"):
            raise EvaluationGateError("sealed match requires independent prefix evidence")
    matches = [a for a in attempts if a["outcome"] == "prefix_match"]
    selected = (
        min(matches, key=lambda a: (len(a["tokens"]), a["source_sha256"], a["attempt_index"]))[
            "attempt_index"
        ]
        if matches
        else None
    )
    if seal["selected"] != selected or (selected is not None and type(seal["selected"]) is not int):
        raise EvaluationGateError("sealed selection is not the deterministic prefix-only choice")
    return seal


def _execute_stage(runtime, source, indices, target, intent_name, request_id, remaining):
    # Keep the idempotency payload stable if a crash occurs after execution but
    # before the CandidateResult is written. A separate inherited phase deadline
    # can tighten retry execution without changing that immutable attempt.
    path = target / intent_name
    if not path.exists():
        write_json(
            target,
            intent_name,
            {
                "limits": asdict(Limits(deadline_ns=min(2_000_000_000, remaining))),
                "request_id": request_id,
                "source_sha256": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
            },
        )
    intent = load_json(path.read_bytes())
    if (
        intent["request_id"] != request_id
        or intent["source_sha256"] != "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    ):
        raise ValueError("attempt intent changed on resume")
    return runtime.run_single(
        source,
        indices,
        request_id=request_id,
        limits=Limits(**intent["limits"]),
        deadline_ns=time.monotonic_ns() + remaining,
    )


def evaluate_foundation(
    checkpoint_path,
    cohort_root,
    protocol_path,
    output,
    *,
    split="development",
    finalization=None,
    device="cpu",
    _hook=None,
):
    """Produce finite diagnostic scores; release qualification is a later gate."""
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.decoder.program_codec import encode_body
    from oeis_learn.evaluation.finalization import (
        validate_finalization,
        final_output,
        exposure_path,
    )
    from oeis_learn.evaluation.readiness import foundation_readiness

    if split not in ("development", "final"):
        raise ValueError("unknown split")
    if device not in ("cpu", "cuda"):
        raise ValueError("explicit cpu or cuda device required")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise HardwareUnavailable("requested GPU unavailable; no CPU fallback")
    cohort_root, output = Path(cohort_root), Path(output)
    protocol = validate_protocol(load_json(Path(protocol_path).read_bytes()))
    cohort = load_cohort(cohort_root)
    checkpoint = load_foundation_checkpoint(checkpoint_path, device="cpu")
    checkpoint_sha = checkpoint.manifest["blob_sha256"]
    provenance = checkpoint.contract["run_provenance"]
    for key, expected in (
        ("benchmark_manifest", cohort["cohort_id"]),
        ("evaluation_protocol", digest(protocol)),
    ):
        if provenance[key] is not None and provenance[key] != expected:
            raise ValueError(f"checkpoint {key} binding mismatch")
    for key, actual in [
        ("checkpoint_sha256", checkpoint_sha),
        ("cohort_sha256", cohort["cohort_id"]),
    ]:
        if key in protocol and protocol[key] != actual:
            raise ValueError(f"protocol {key} mismatch")
    lock = None
    if split == "final":
        if finalization is None:
            raise ValueError("final evaluation requires decision lock")
        try:
            lock = validate_finalization(finalization, checkpoint_path, cohort_root, protocol_path)
        except ValueError as exc:
            raise EvaluationGateError(str(exc)) from exc
        if output.resolve() != final_output(lock).resolve():
            raise ValueError("final output is fixed by decision lock")
    elif finalization is not None:
        raise ValueError("decision lock is only for final evaluation")
    identity = {
        "schema_version": "foundation/v1",
        "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "cohort_id": cohort["cohort_id"],
        "protocol_sha256": digest(protocol),
        "split": split,
        "finalization_id": lock["finalization_id"] if lock else None,
        "runtime_sha256": digest(runtime_identity()),
        "device": device,
        "evaluator_sha256": digest(
            {
                name: compute_file_hash(Path(__file__).parent / name)
                for name in (
                    "foundation_synthesis.py",
                    "foundation_cohort.py",
                    "checkpoint.py",
                    "finalization.py",
                )
            }
        ),
    }
    evaluation_id = digest(identity)
    write_json(output, "evaluation.json", dict(identity, evaluation_id=evaluation_id))
    groups = [g for g in cohort["groups"] if g["partition"] == split]
    if not groups:
        raise ValueError("requested split is empty")
    # Fail closed before generation if any target has lost a seal after exposure.
    for group in groups:
        target = output / "targets" / group["group_id"][7:]
        if (
            (target / "hidden-exposure.json").exists()
            or (lock and exposure_path(lock, group["group_id"]).exists())
        ) and not (target / "seal.json").exists():
            raise EvaluationGateError(
                "hidden exposure without committed seal invalidates evaluation"
            )

    def hook(event, target):
        if _hook:
            _hook(event, target)

    results, seals = [], []
    # The loaded parent modules are not passed to the spawned generator.
    run_id = checkpoint.manifest["run_id"]
    readiness = foundation_readiness(checkpoint, software_complete=True)
    del checkpoint
    with ExitStack() as stack:
        runtime = stack.enter_context(
            FoundationRunner(output / "execution", backend="python_wasmtime")
        )
        model = None
        for group in groups:
            target = output / "targets" / group["group_id"][7:]
            target.mkdir(parents=True, exist_ok=True)
            seal_path = target / "seal.json"
            prompt = read_ref(cohort_root, group["prompt"])
            prompt_hash = digest(prompt["observed_terms"])
            if seal_path.exists():
                seal = _load_seal(
                    seal_path, evaluation_id, protocol, prompt, group["group_id"], checkpoint_sha
                )
            else:
                if model is None:
                    model = stack.enter_context(ModelProcess(checkpoint_path, device=device))
                    if model.checkpoint_sha256 != checkpoint_sha:
                        raise ValueError("loaded model checkpoint mismatch")
                elif model.process is None:
                    model.start()
                attempts = []
                complete = True
                for index in range(protocol["attempts"]):
                    saved = target / f"attempt-{index:02}.json"
                    if saved.exists():
                        attempts.append(load_json(saved.read_bytes()))
                        continue
                    remaining = _remaining(
                        target, "prefix-budget.json", protocol["prefix_budget_ns"]
                    )
                    if remaining <= 0:
                        complete = False
                        break
                    seed = attempt_seed(protocol, prompt["observed_terms"], index)
                    generation_start = time.monotonic_ns()
                    proposal = model.generate(
                        prompt,
                        seed=seed,
                        temperature=0 if index == 0 else protocol["temperature"],
                        max_body_tokens=protocol["max_body_tokens"],
                        deadline_ns=time.monotonic_ns() + remaining,
                    )
                    attempt = {
                        "attempt_index": index,
                        "seed": seed,
                        **proposal,
                        "source_sha256": None,
                        "prefix_result": None,
                        "generation_elapsed_ns": time.monotonic_ns() - generation_start,
                        "generation_peak_memory": {
                            "state": "unavailable",
                            "value": None,
                            "unit": "bytes",
                            "source": None,
                            "reason": "not instrumented in this software gate",
                        },
                    }
                    if proposal["source"] is not None:
                        attempt["source_sha256"] = (
                            "sha256:" + hashlib.sha256(proposal["source"].encode()).hexdigest()
                        )
                        tokens = encode_body(proposal["source"])
                        if tokens != proposal["tokens"]:
                            raise RuntimeError("model source/token mismatch")
                        remaining = _remaining(
                            target, "prefix-budget.json", protocol["prefix_budget_ns"]
                        )
                        if remaining <= 0:
                            attempt.update(
                                outcome="execution_limit", reason="prefix_phase_deadline"
                            )
                        else:
                            evidence = _execute_stage(
                                runtime,
                                proposal["source"],
                                range(20),
                                target,
                                f"prefix-{index:02}.intent.json",
                                f"{group['group_id']}:{index}:prefix",
                                remaining,
                            )
                            result = verify(
                                evidence,
                                prompt["observed_terms"],
                                "prefix",
                                run_id=run_id,
                                attempt_index=index,
                                prompt_sha256=prompt_hash,
                                checkpoint_sha256=checkpoint_sha,
                                purpose="model",
                            )
                            attempt.update(
                                source_sha256=result["source_sha256"],
                                prefix_result=result,
                                outcome=result["outcome"],
                                reason=result["reason"],
                            )
                    write_json(target, saved.name, attempt)
                    attempts.append(attempt)
                    hook("after_attempt", target)
                matches = [a for a in attempts if a["outcome"] == "prefix_match"]
                selected = (
                    min(
                        matches,
                        key=lambda a: (len(a["tokens"]), a["source_sha256"], a["attempt_index"]),
                    )["attempt_index"]
                    if matches
                    else None
                )
                for attempt in attempts:
                    if attempt["prefix_result"] is not None:
                        attempt["prefix_result"]["selected"] = attempt["attempt_index"] == selected
                seal = {
                    "schema_version": "foundation/v1",
                    "evaluation_id": evaluation_id,
                    "group_id": group["group_id"],
                    "prompt_sha256": prompt_hash,
                    "checkpoint_sha256": checkpoint_sha,
                    "finalization_id": identity["finalization_id"],
                    "attempts": attempts,
                    "selected": selected,
                    "generation_complete": complete,
                    "selector": protocol["selector"],
                }
                seal["seal_id"] = digest(seal)
                write_json(target, "seal.json", seal)
                _load_seal(
                    seal_path, evaluation_id, protocol, prompt, group["group_id"], checkpoint_sha
                )
                hook("after_seal", target)
            seals.append(seal["seal_id"])
            if (target / "score.json").exists():
                score = load_json((target / "score.json").read_bytes())
                if score["seal_id"] != seal["seal_id"] or score["score_id"] != digest(
                    score, "score_id"
                ):
                    raise EvaluationGateError("score/seal mismatch")
                results.append(score)
                continue
            _remaining(target, "full-budget.json", protocol["full_budget_ns"])
            exposure = {"evaluation_id": evaluation_id, "seal_id": seal["seal_id"]}
            if lock:
                anchor = exposure_path(lock, group["group_id"])
                write_json(anchor.parent, anchor.name, exposure)
            write_json(target, "hidden-exposure.json", exposure)
            hook("after_exposure", target)
            truth = read_ref(cohort_root, group["truth"])
            values = truth.get("values")
            if (
                not isinstance(values, list)
                or len(values) != 100
                or values[:20] != prompt["observed_terms"]
                or truth.get("record_id") != group["representative"]
                or truth.get("indices")
                != list(range(truth["first_index"], truth["first_index"] + 100))
            ):
                raise ValueError("missing/incompatible exact100 benchmark truth")
            for value in values:
                parse_integer_text(value, I256_MIN, I256_MAX, "truth")
            full = []
            order = sorted(
                [a for a in seal["attempts"] if a["outcome"] == "prefix_match"],
                key=lambda a: (a["attempt_index"] != seal["selected"], a["attempt_index"]),
            )
            for attempt in order:
                index = attempt["attempt_index"]
                name = f"full-{index:02}.json"
                path = target / name
                if path.exists():
                    record = load_json(path.read_bytes())
                else:
                    remaining = _remaining(target, "full-budget.json", protocol["full_budget_ns"])
                    if remaining <= 0:
                        record = {
                            "attempt_index": index,
                            "outcome": "execution_limit",
                            "reason": "full_phase_deadline",
                            "selected": index == seal["selected"],
                        }
                    else:
                        evidence = _execute_stage(
                            runtime,
                            attempt["source"],
                            range(100),
                            target,
                            f"full-{index:02}.intent.json",
                            f"{group['group_id']}:{index}:full",
                            remaining,
                        )
                        record = verify(
                            evidence,
                            values,
                            "full",
                            run_id=run_id,
                            attempt_index=index,
                            prompt_sha256=prompt_hash,
                            checkpoint_sha256=checkpoint_sha,
                            purpose="model",
                            selected=index == seal["selected"],
                        )
                    write_json(target, name, record)
                full.append(record)
            score = {
                "evaluation_id": evaluation_id,
                "group_id": group["group_id"],
                "seal_id": seal["seal_id"],
                "full_results": full,
                "success_any": any(r["outcome"] == "full_horizon_match" for r in full),
                "success_top1": any(
                    r["outcome"] == "full_horizon_match" and r["selected"] for r in full
                ),
                "prefix_outcomes": dict(Counter(a["outcome"] for a in seal["attempts"])),
                "generation_complete": seal["generation_complete"],
            }
            score["score_id"] = digest(score)
            write_json(target, "score.json", score)
            results.append(score)
            hook("after_score", target)
    n = len(groups)
    report = {
        "schema_version": "foundation/v1",
        "command": "evaluate",
        "status": "complete",
        "software_gate": "G2",
        "evaluation_id": evaluation_id,
        "checkpoint_sha256": checkpoint_sha,
        "cohort_id": cohort["cohort_id"],
        "protocol_sha256": digest(protocol),
        "split": split,
        "finalization_id": identity["finalization_id"],
        "seal_ids": seals,
        "score_ids": [r["score_id"] for r in results],
        "N": n,
        "success_any": sum(r["success_any"] for r in results),
        "success_top1": sum(r["success_top1"] for r in results),
        "prefix_outcomes": dict(sum((Counter(r["prefix_outcomes"]) for r in results), Counter())),
        "full_outcomes": dict(Counter(x["outcome"] for r in results for x in r["full_results"])),
        "qualified": readiness["qualified"],
        "qualification": readiness["qualification"],
        "readiness": readiness,
        "proof_status": "not_claimed",
        "novelty_status": "not_claimed",
    }
    report["rates"] = {
        "success_any": report["success_any"] / n,
        "success_top1": report["success_top1"] / n,
    }
    report["report_id"] = digest(report)
    write_json(output, "reports/evaluation.json", report)
    return report
