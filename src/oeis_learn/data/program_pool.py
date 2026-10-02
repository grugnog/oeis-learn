"""Bounded offline generic pools, ordered journals and a narrow trainer view."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time

from oeis_learn.data.generic_programs import GenericSampler, load_sampler_config
from oeis_learn.data.program_admission import (
    AdmissionService,
    AdmissionGateError,
    load_membership,
    validate_program,
    validate_sample,
)
from oeis_learn.decoder.program_codec import decode_body, encode_body
from oeis_learn.evaluation.foundation_cohort import exact_keys, read_ref, write_json
from oeis_learn.experiments.artifacts import (
    compute_canonical_digest as digest,
    compute_file_hash,
    load_json,
)
from oeis_learn.experiments.models import parse_integer_text, check_digest
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX, profile_digests
from oeis_learn.sandbox.pipeline import digest_bytes, Limits, runtime_identity
from oeis_learn.sandbox.worker_pool import WorkerPool


def ordered_results(futures):
    """Completion speed never changes the sample-counter publication order."""
    for counter in sorted(futures):
        yield counter, futures[counter].result()


def representative_key(record):
    return len(record["body_tokens"]), record["source_sha256"], record["sample_counter"]


def _clock():
    return {
        "monotonic_ns": time.monotonic_ns(),
        "utc_ns": time.time_ns(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def _elapsed(start):
    now = _clock()
    key = "monotonic_ns" if now["boot_id"] == start["boot_id"] else "utc_ns"
    elapsed = now[key] - start[key]
    if elapsed < 0 or now["utc_ns"] < start["utc_ns"]:
        raise ValueError("backward pool clock; cannot refund spent budget")
    return elapsed


def _implementation():
    root = Path(__file__).parent
    return {
        name: compute_file_hash(root / name)
        for name in ("generic_programs.py", "program_admission.py", "program_pool.py")
    }


@dataclass(frozen=True)
class TrainerPool:
    pool_id: str
    view_id: str
    examples: tuple[dict, ...]


def load_trainer_view(root, *, expected_view_id):
    """Learner-side loader: only a controller-approved, content-pinned view."""
    check_digest(expected_view_id, "expected trainer view")
    root = Path(root)
    manifest = load_json((root / "manifest.json").read_bytes())
    exact_keys(
        manifest,
        ("schema_version", "kind", "codec_profile", "count", "examples", "view_id"),
        "trainer view manifest",
    )
    if (
        manifest["schema_version"] != "foundation/v1"
        or manifest["kind"] != "trainer_view"
        or manifest["view_id"] != digest(manifest, "view_id")
        or manifest["view_id"] != expected_view_id
        or manifest["codec_profile"] != profile_digests()["codec"]
    ):
        raise ValueError("unverified or incompatible trainer view")
    examples = read_ref(root, manifest["examples"])
    if (
        type(manifest["count"]) is not int
        or not isinstance(examples, list)
        or not 1 <= len(examples) == manifest["count"] <= 64
    ):
        raise ValueError("missing trainer examples")
    ids = set()
    for example in examples:
        exact_keys(
            example,
            ("program_id", "codec_profile", "visible_terms", "body_tokens"),
            "trainer example",
        )
        check_digest(example["program_id"], "program identity")
        if (
            example["program_id"] in ids
            or example["codec_profile"] != manifest["codec_profile"]
            or not isinstance(example["visible_terms"], list)
            or len(example["visible_terms"]) != 20
            or not isinstance(example["body_tokens"], list)
            or any(type(t) is not int for t in example["body_tokens"])
        ):
            raise ValueError("invalid trainer example")
        ids.add(example["program_id"])
        for value in example["visible_terms"]:
            parse_integer_text(value, I256_MIN, I256_MAX, "conditioning value")
        if encode_body(decode_body(example["body_tokens"])) != example["body_tokens"]:
            raise ValueError("trainer token roundtrip failed")
    return tuple(examples)


class _State:
    def __init__(self):
        self.cursor = 0
        self.sources = set()
        self.winners = {}
        self.first_seen = {}
        self.counts, self.reasons = Counter(), Counter()
        self.operators, self.statements, self.leaves, self.constants = (Counter() for _ in range(4))
        self.emitted = Counter()

    def add(self, sample, decision, record, record_ref, decision_ref):
        if sample["sample_counter"] != self.cursor or decision["sample_counter"] != self.cursor:
            raise ValueError("nonconsecutive sample journal")
        self.cursor += 1
        self.sources.add(digest_bytes(sample["canonical_source"].encode()))
        self.counts[decision["status"]] += 1
        if decision["reason"]:
            self.reasons[decision["reason"]] += 1
        for counter, name in (
            (self.operators, "operators"),
            (self.statements, "statement_kinds"),
            (self.leaves, "leaves"),
            (self.constants, "constants"),
            (self.emitted, "emitted_operators"),
        ):
            counter.update(sample["statistics"][name])
        if record is not None:
            outputs = record.pop("_outputs")
            fingerprint = digest(outputs)
            self.first_seen.setdefault(fingerprint, sample["sample_counter"])
            item = {
                "record": record,
                "record_ref": record_ref,
                "decision_ref": decision_ref,
                "output_fingerprint": fingerprint,
                "constant": len(set(outputs)) == 1,
            }
            previous = self.winners.get(fingerprint)
            if previous is None or representative_key(record) < representative_key(
                previous["record"]
            ):
                self.winners[fingerprint] = item

    def selected(self, count):
        classes = sorted(self.winners, key=lambda x: self.first_seen[x])[:count]
        return sorted(
            (self.winners[x] for x in classes), key=lambda x: x["record"]["sample_counter"]
        )


def _load_attempt(root, ref, sampler, membership):
    attempt = read_ref(root, ref)
    exact_keys(attempt, ("sample", "decision", "program"), "attempt journal")
    sample = read_ref(root, attempt["sample"])
    if digest(sample) != digest(sampler.sample(sample["sample_counter"])):
        raise ValueError("sample journal provenance changed")
    decision = read_ref(root, attempt["decision"])
    if (
        decision["decision_id"] != digest(decision, "decision_id")
        or decision["membership_sha256"] != membership
        or decision["sample_counter"] != sample["sample_counter"]
        or decision["status"] not in ("admitted", "rejected")
    ):
        raise ValueError("invalid admission journal")
    record = None
    if decision["status"] == "admitted":
        validate_program(root, attempt["program"], attempt["decision"], sampler, membership)
        record = read_ref(root, attempt["program"])
        record["_outputs"] = read_ref(root, record["outputs_ref"])["values"]
    elif attempt["program"] is not None or not decision["reason"]:
        raise ValueError("invalid rejected attempt")
    return sample, decision, record, attempt["program"], attempt["decision"]


def _replay_chunks(root, refs, sampler, membership, build_id):
    state, previous = _State(), None
    for ref in refs:
        chunk = read_ref(root, ref)
        exact_keys(
            chunk,
            (
                "build_id",
                "start",
                "stop",
                "attempts",
                "batch",
                "previous_chunk_id",
                "rng_identity",
                "dedup_sources",
                "dedup_outputs",
                "chunk_id",
            ),
            "builder checkpoint",
        )
        if (
            chunk["chunk_id"] != digest(chunk, "chunk_id")
            or chunk["build_id"] != build_id
            or chunk["previous_chunk_id"] != previous
            or chunk["start"] != state.cursor
            or chunk["rng_identity"] != sampler.rng_identity
            or not 1 <= len(chunk["attempts"]) <= sampler.config["chunk_size"]
        ):
            raise ValueError("invalid builder checkpoint chain")
        batch = read_ref(root, chunk["batch"])
        if batch != {"build_id": build_id, "start": chunk["start"], "stop": chunk["stop"]}:
            raise ValueError("chunk does not match issued batch")
        for attempt in chunk["attempts"]:
            state.add(*_load_attempt(root, attempt, sampler, membership))
        if (
            chunk["stop"] != state.cursor
            or chunk["dedup_sources"] != digest(sorted(state.sources))
            or chunk["dedup_outputs"]
            != digest({key: x["record"]["program_id"] for key, x in state.winners.items()})
        ):
            raise ValueError("builder cursor/dedup checkpoint mismatch")
        previous = chunk["chunk_id"]
    return state


def load_pool(root, *, expected_pool_id):
    """Controller verifies private admission archive before supplying the learner view."""
    root = Path(root)
    check_digest(expected_pool_id, "expected pool")
    manifest = load_json((root / "manifest.json").read_bytes())
    exact_keys(
        manifest,
        (
            "schema_version",
            "kind",
            "build",
            "chunks",
            "entries",
            "trainer_view",
            "trainer_view_id",
            "summary",
            "pool_id",
            "completion_elapsed_ns",
        ),
        "pool manifest",
    )
    if (
        manifest["schema_version"] != "foundation/v1"
        or manifest["kind"] != "generic_pool"
        or manifest["pool_id"] != expected_pool_id
        or manifest["pool_id"] != digest(manifest, "pool_id")
    ):
        raise ValueError("unverified/incompatible pool")
    build = read_ref(root, manifest["build"])
    exact_keys(
        build,
        (
            "schema_version",
            "kind",
            "config",
            "generator_revision",
            "rng_identity",
            "membership_sha256",
            "runtime_sha256",
            "profiles",
            "implementation",
            "build_id",
        ),
        "pool build",
    )
    if (
        build["schema_version"] != "foundation/v1"
        or build["kind"] != "pool_build"
        or build["profiles"] != profile_digests()
    ):
        raise ValueError("incompatible pool build")
    sampler = GenericSampler(build["config"])
    if (
        build["build_id"] != digest(build, "build_id")
        or build["generator_revision"] != sampler.revision
        or build["rng_identity"] != sampler.rng_identity
        or build["runtime_sha256"] != digest(runtime_identity())
        or build["implementation"] != _implementation()
    ):
        raise ValueError("pool configuration/generator/runtime mismatch")
    state = _replay_chunks(
        root, manifest["chunks"], sampler, build["membership_sha256"], build["build_id"]
    )
    if (
        type(manifest["completion_elapsed_ns"]) is not int
        or not 0 <= manifest["completion_elapsed_ns"] < sampler.config["budget_ns"]
    ):
        raise ValueError("pool was not committed inside its frozen budget")
    chosen = state.selected(sampler.config["target_count"])
    expected = [
        {
            "program": x["record_ref"],
            "decision": x["decision_ref"],
            "output_fingerprint": x["output_fingerprint"],
        }
        for x in chosen
    ]
    if len(chosen) != sampler.config["target_count"] or manifest["entries"] != expected:
        raise ValueError("pool is incomplete or dedup/order changed")
    if manifest["summary"] != _summary(state, sampler.config["target_count"]):
        raise ValueError("pool summary differs from admission evidence")
    view = read_ref(root, manifest["trainer_view"])
    if view["view_id"] != manifest["trainer_view_id"]:
        raise ValueError("pool/trainer identity mismatch")
    examples = load_trainer_view(root / "trainer", expected_view_id=manifest["trainer_view_id"])
    expected_examples = tuple(
        validate_program(
            root, x["record_ref"], x["decision_ref"], sampler, build["membership_sha256"]
        )
        for x in chosen
    )
    if examples != expected_examples:
        raise ValueError("trainer view differs from admitted pool")
    return TrainerPool(manifest["pool_id"], view["view_id"], examples)


def build_pool(config_path, cohort_root, output, *, _hook=None):
    """Process fixed eight-sample chunks; never tune priors to improve yield."""
    config = load_sampler_config(config_path)
    sampler = GenericSampler(config)
    reserved, membership = load_membership(cohort_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    build = {
        "schema_version": "foundation/v1",
        "kind": "pool_build",
        "config": config,
        "generator_revision": sampler.revision,
        "rng_identity": sampler.rng_identity,
        "membership_sha256": membership,
        "runtime_sha256": digest(runtime_identity()),
        "profiles": profile_digests(),
        "implementation": _implementation(),
    }
    build["build_id"] = digest(build)
    build_ref = write_json(output, "build.json", build)
    report_path = output / "reports/pool.json"
    if (output / "reports/gate-failure.json").exists():
        raise AdmissionGateError("previous independent-evidence gate failure")
    if report_path.exists():
        report = load_json(report_path.read_bytes())
        if (
            report.get("report_id") != digest(report, "report_id")
            or report["build_id"] != build["build_id"]
        ):
            raise ValueError("pool report identity mismatch")
        if report["status"] == "complete":
            load_pool(output, expected_pool_id=report["pool_id"])
        return report
    if (output / "manifest.json").exists():
        # Manifest publication is the commit point. A crash before the report
        # cannot reopen generation or turn a completed pool into a shortfall.
        manifest = load_json((output / "manifest.json").read_bytes())
        load_pool(output, expected_pool_id=manifest["pool_id"])
        report = _complete_report(output, build, manifest)
        write_json(output, "reports/pool.json", report)
        return report
    timer = output / "budget-start.json"
    if not timer.exists():
        write_json(output, timer.name, _clock())
    start = load_json(timer.read_bytes())
    chunks = [
        {"path": str(p.relative_to(output)), "sha256": compute_file_hash(p)}
        for p in sorted((output / "chunks").glob("*.json"))
    ]
    state = _replay_chunks(output, chunks, sampler, membership, build["build_id"])
    service = AdmissionService(output, sampler, reserved, membership)
    reason = None

    def hook(event):
        if _hook:
            _hook(event, output)

    try:
        needs_work = (
            len(state.winners) < config["target_count"]
            and state.cursor < config["max_attempts"]
            and _elapsed(start) < config["budget_ns"]
        )
        worker_context = (
            WorkerPool(output / "execution", workers=config["workers"])
            if needs_work
            else nullcontext()
        )
        with worker_context as runtime:
            while (
                len(state.winners) < config["target_count"]
                and state.cursor < config["max_attempts"]
            ):
                remaining = config["budget_ns"] - _elapsed(start)
                if remaining <= 0:
                    reason = "wall_budget_exhausted"
                    break
                begin = state.cursor
                stop = min(begin + config["chunk_size"], config["max_attempts"])
                batch_ref = write_json(
                    output,
                    f"batches/{begin:05}.json",
                    {"build_id": build["build_id"], "start": begin, "stop": stop},
                )
                samples, futures, duplicates, seen = {}, {}, set(), set(state.sources)
                deadline = time.monotonic_ns() + remaining
                for counter in range(begin, stop):
                    sample = samples[counter] = sampler.sample(counter)
                    write_json(output, f"samples/{counter:05}.json", sample)
                    source_hash = digest_bytes(sample["canonical_source"].encode())
                    if source_hash in seen:
                        duplicates.add(counter)
                    seen.add(source_hash)
                    if validate_sample(sample, sampler) is None and counter not in duplicates:
                        futures[counter] = runtime.submit(
                            sample["canonical_source"],
                            range(100),
                            request_id=digest([build["build_id"], counter]),
                            limits=Limits(),
                            deadline_ns=deadline,
                        )
                results = dict(ordered_results(futures))
                attempts = []
                for counter in range(begin, stop):
                    sample = samples[counter]
                    sample_ref = write_json(output, f"samples/{counter:05}.json", sample)
                    admitted = service.consider(
                        sample, results.get(counter), duplicate=counter in duplicates
                    )
                    attempt_ref = write_json(
                        output,
                        f"attempts/{counter:05}.json",
                        {
                            "sample": sample_ref,
                            "decision": admitted.decision_ref,
                            "program": admitted.record_ref,
                        },
                    )
                    attempts.append(attempt_ref)
                    record = admitted.record
                    if record is not None:
                        record = {**record, "_outputs": results[counter].outputs}
                    state.add(
                        sample,
                        admitted.decision,
                        record,
                        admitted.record_ref,
                        admitted.decision_ref,
                    )
                    hook("after_attempt")
                previous = read_ref(output, chunks[-1])["chunk_id"] if chunks else None
                chunk = {
                    "build_id": build["build_id"],
                    "start": begin,
                    "stop": stop,
                    "attempts": attempts,
                    "batch": batch_ref,
                    "previous_chunk_id": previous,
                    "rng_identity": sampler.rng_identity,
                    "dedup_sources": digest(sorted(state.sources)),
                    "dedup_outputs": digest(
                        {key: x["record"]["program_id"] for key, x in state.winners.items()}
                    ),
                }
                chunk["chunk_id"] = digest(chunk)
                chunks.append(write_json(output, f"chunks/{begin:05}.json", chunk))
                hook("after_chunk")
    except AdmissionGateError:
        # Independent disagreement is a correctness gate, not low generic yield.
        write_json(
            output,
            "reports/gate-failure.json",
            {"build_id": build["build_id"], "reason": "invalid_independent_evidence"},
        )
        raise
    issued = _issued_stop(output, build["build_id"], config)
    # An expired interrupted chunk still consumed issued attempt IDs. Recover
    # any durable decisions without executing unfinished requests or refunding
    # their counters; unknown outcomes remain explicitly pending.
    while state.cursor < issued:
        path = output / f"attempts/{state.cursor:05}.json"
        if not path.exists():
            break
        ref = {"path": str(path.relative_to(output)), "sha256": compute_file_hash(path)}
        state.add(*_load_attempt(output, ref, sampler, membership))
    selected = state.selected(config["target_count"])
    summary = _summary(state, config["target_count"], issued=issued)
    report = {
        "schema_version": "foundation/v1",
        "command": "build-pool",
        "gate": "G3",
        "build_id": build["build_id"],
        "status": "shortfall",
        "purpose": "diagnostic",
        "qualified": False,
        "proof_status": "not_claimed",
        "pool_id": None,
        "requested": config["target_count"],
        "published": 0,
        "summary": summary,
        "reason": reason or "attempt_budget_exhausted",
        "elapsed_ns": _elapsed(start),
        "result_path": str(output / "reports/pool.json"),
    }
    if len(selected) == config["target_count"] and report["elapsed_ns"] < config["budget_ns"]:
        examples = [
            validate_program(output, x["record_ref"], x["decision_ref"], sampler, membership)
            for x in selected
        ]
        hook("before_publish")
        if _elapsed(start) < config["budget_ns"]:
            example_ref = write_json(output / "trainer", "examples.json", examples)
            view = {
                "schema_version": "foundation/v1",
                "kind": "trainer_view",
                "codec_profile": profile_digests()["codec"],
                "count": len(examples),
                "examples": example_ref,
            }
            view["view_id"] = digest(view)
            view_ref = write_json(output, "trainer/manifest.json", view)
            manifest = {
                "schema_version": "foundation/v1",
                "kind": "generic_pool",
                "build": build_ref,
                "chunks": chunks,
                "entries": [
                    {
                        "program": x["record_ref"],
                        "decision": x["decision_ref"],
                        "output_fingerprint": x["output_fingerprint"],
                    }
                    for x in selected
                ],
                "trainer_view": view_ref,
                "trainer_view_id": view["view_id"],
                "summary": summary,
                "completion_elapsed_ns": _elapsed(start),
            }
            manifest["pool_id"] = digest(manifest)
            write_json(output, "manifest.json", manifest)
            hook("after_publish")
            report = _complete_report(output, build, manifest)
    if report["status"] != "complete" and _elapsed(start) >= config["budget_ns"]:
        report["reason"] = "wall_budget_exhausted"
    if report["status"] != "complete":
        report["elapsed_ns"] = _elapsed(start)
        report["report_id"] = digest(report)
    write_json(output, "reports/pool.json", report)
    return report


def _complete_report(output, build, manifest):
    report = {
        "schema_version": "foundation/v1",
        "command": "build-pool",
        "gate": "G3",
        "build_id": build["build_id"],
        "status": "complete",
        "purpose": "diagnostic",
        "qualified": False,
        "proof_status": "not_claimed",
        "pool_id": manifest["pool_id"],
        "requested": build["config"]["target_count"],
        "published": len(manifest["entries"]),
        "summary": manifest["summary"],
        "reason": None,
        "elapsed_ns": manifest["completion_elapsed_ns"],
        "result_path": str(output / "reports/pool.json"),
    }
    report["report_id"] = digest(report)
    return report


def _summary(state, target_count, *, issued=None):
    issued = state.cursor if issued is None else issued
    selected = state.selected(target_count)
    return {
        "attempted": issued,
        "decided_attempts": state.cursor,
        "pending_attempts": issued - state.cursor,
        "admitted": state.counts["admitted"],
        "rejected": state.counts["rejected"],
        "rejection_reasons": dict(state.reasons),
        "unique_sources": len(state.sources),
        "unique_outputs": len(state.winners),
        "output_duplicates": state.counts["admitted"] - len(state.winners),
        "constant_outputs": sum(x["constant"] for x in selected),
        "constant_fraction": sum(x["constant"] for x in selected) / len(selected)
        if selected
        else None,
        "sampled_operators": dict(state.operators),
        "emitted_operators": dict(state.emitted),
        "sampled_statement_kinds": dict(state.statements),
        "sampled_leaves": dict(state.leaves),
        "sampled_constant_bins": dict(state.constants),
    }


def _issued_stop(root, build_id, config):
    cursor = 0
    for path in sorted((root / "batches").glob("*.json")):
        ref = {"path": str(path.relative_to(root)), "sha256": compute_file_hash(path)}
        batch = read_ref(root, ref)
        stop = min(cursor + config["chunk_size"], config["max_attempts"])
        if (
            path.name != f"{cursor:05}.json"
            or cursor >= stop
            or batch != {"build_id": build_id, "start": cursor, "stop": stop}
        ):
            raise ValueError("invalid issued-batch journal")
        cursor = stop
    return cursor
