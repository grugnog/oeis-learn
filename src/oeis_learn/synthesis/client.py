"""Bounded transport and independent admission; no Python fitting algorithms."""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
import uuid

from .contracts import ResourceProfile, identity
from .admission import VisiblePrompt, admit, reference_identity, source_identity

MAX_FRAME = 4 * 1024**2


def encoded(value):
    parts = []
    size = 0
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(
        value
    ):
        chunk = part.encode()
        size += len(chunk)
        if size > MAX_FRAME:
            raise ValueError("frame exceeds 4 MiB")
        parts.append(chunk)
    return b"".join(parts)


def frame(value):
    body = encoded(value)
    return struct.pack("<I", len(body)) + body


class FrameReader:
    def __init__(self):
        self.buffer = bytearray()
        self.length = None

    def feed(self, data):
        result = []
        # Inspect each header before appending its body. The caller reads bounded chunks.
        at = 0
        while at < len(data):
            needed = (4 if self.length is None else self.length) - len(self.buffer)
            take = min(needed, len(data) - at)
            self.buffer.extend(data[at : at + take])
            at += take
            if self.length is None and len(self.buffer) == 4:
                self.length = struct.unpack("<I", self.buffer)[0]
                self.buffer.clear()
                if not 0 < self.length <= MAX_FRAME:
                    raise ValueError("invalid frame length")
            if self.length is not None and len(self.buffer) == self.length:
                value = json.loads(
                    self.buffer, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
                )
                if not isinstance(value, dict):
                    raise ValueError("object frame required")
                result.append(value)
                self.buffer.clear()
                self.length = None
        return result


def batches(candidates, *, max_count=32, max_bytes=MAX_FRAME - 16384):
    if not 1 <= max_count <= 32 or not 1 <= max_bytes < MAX_FRAME:
        raise ValueError("invalid batch limits")
    batch = []
    size = 32
    seen = set()
    for candidate in candidates:
        key = candidate["candidate_id"]
        if not isinstance(key, str) or key in seen:
            raise ValueError("duplicate candidate ID")
        seen.add(key)
        charge = len(encoded(candidate)) + 1
        if charge + 32 > max_bytes:
            raise ValueError("candidate exceeds frame budget")
        if batch and (len(batch) == max_count or size + charge > max_bytes):
            yield batch
            batch = []
            size = 32
        batch.append(candidate)
        size += charge
    if batch:
        yield batch


class SynthesisClient:
    def __init__(
        self,
        contract_sha256,
        *,
        worker=None,
        limits=ResourceProfile(),
        service_limits=None,
        cache_root=None,
    ):
        from .worker_pool import WorkerPool, RoleCaches
        from .service_policy import ServiceLimits

        self.contract = contract_sha256
        self.limits = limits
        self.service_limits = service_limits or ServiceLimits()
        self.worker = worker or WorkerPool(
            contract_sha256=contract_sha256, resources=limits, limits=self.service_limits
        )
        self.caches = RoleCaches(self.service_limits, cache_root)
        self.reference_sha256 = reference_identity()

    def close(self):
        self.worker.close()
        self.caches.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def runtime_identity(self, profile):
        binary = Path(self.worker.command[0])
        guest = Path(
            os.environ.get("OEIS_BIGINT_WASM", "/opt/oeis-build-info/bigint-dependency-smoke.wasm")
        )
        with binary.open("rb") as handle:
            binary_hash = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
        return identity(
            dict(
                binary_sha256=binary_hash,
                guest_sha256="sha256:" + hashlib.sha256(guest.read_bytes()).hexdigest(),
                profile=profile,
            )
        )

    def request(
        self,
        operation,
        candidates,
        indices,
        values=(),
        *,
        profile="bigint-body/v1",
        deadline=None,
        seed=20261016,
    ):
        deadline = time.monotonic() + 30 if deadline is None else deadline
        rows = []
        for candidate in candidates:
            source = candidate["source"]
            row = dict(
                candidate_id=candidate["candidate_id"],
                original_source=source,
                original_source_sha256=source_identity(source),
                codec_profile="bigint-codec/v1" if profile == "bigint-body/v1" else profile,
                parameter_ids=list(dict.fromkeys(re.findall(r"param\.get\s+(p[0-7])", source))),
                abstraction_policy=candidate.get("abstraction_policy", "none"),
            )
            if "evidence" in candidate:
                row["evidence"] = candidate["evidence"]
            rows.append(row)
        results = []
        unstarted = []
        complete = True
        for batch in batches(rows):
            request = dict(
                protocol="synthesis-worker/v1",
                request_id="req-" + uuid.uuid4().hex,
                operation=operation,
                worker_generation=self.worker.generation,
                contract_sha256=self.contract,
                language_profile=profile,
                resource_profile="bigint-resources/v1",
                solver_profile="native-fitting/v1",
                candidates=batch,
                visible_indices=list(indices),
                visible_values=list(values) if operation != "execute" else ["0"] * len(indices),
                remaining_budget=dict(
                    wall_time_ms=max(1, min(60000, int((deadline - time.monotonic()) * 1000))),
                    attempts=len(batch),
                ),
                seed=seed,
            )
            response = self.worker.call(request, deadline=deadline)
            results.extend(response["results"])
            unstarted.extend(response["unstarted"])
            complete &= response["complete"]
        return dict(results=results, unstarted=unstarted, complete=complete)

    def execute(self, source, indices, *, profile="bigint-body/v1", deadline=None):
        response = self.request(
            "execute",
            [dict(candidate_id="execute", source=source)],
            indices,
            profile=profile,
            deadline=deadline,
        )
        return (
            response["results"][0]
            if response["results"]
            else dict(outcome="resource_exhausted", values=[], reason="worker incomplete")
        )

    def fit(
        self,
        source,
        prompt,
        *,
        profile="bigint-body/v1",
        policy="none",
        deadline=None,
        seed=20261016,
    ):
        if not isinstance(prompt, VisiblePrompt):
            raise ValueError("visible-only prompt required")
        deadline = time.monotonic() + 4 if deadline is None else deadline
        runtime = self.runtime_identity(profile)
        key = identity(
            dict(
                source=source,
                prompt=prompt.sha256,
                resources=self.limits.identity,
                runtime=runtime,
                reference=self.reference_sha256,
                profile=profile,
                policy=policy,
                solver="native-fitting/v1",
                seed=seed,
            )
        )
        native = self.caches.get(key)
        if native is None:
            response = self.request(
                "fit",
                [dict(candidate_id="fit", source=source, abstraction_policy=policy)],
                prompt.indices,
                prompt.values,
                profile=profile,
                deadline=min(
                    time.monotonic() + 2,
                    max(time.monotonic(), deadline - min(2, max(0, deadline - time.monotonic()))),
                ),
                seed=seed,
            )
            native = (
                response["results"][0]
                if response["results"]
                else dict(
                    outcome="resource_exhausted", assignments=[], reason="incomplete worker result"
                )
            )
        verified = []
        for item in native.get("assignments", [])[:8]:
            if item.get("original_source_sha256") != source_identity(source):
                return dict(
                    outcome="verification_failure",
                    native=native,
                    admitted=[],
                    reason="original source mismatch",
                )
            result = admit(
                item, prompt, runtime_sha256=runtime, limits=self.limits, deadline=deadline
            )
            if result.outcome == "verified_solution":
                verified.append(
                    dict(item, verification=result.verification, outcome="verified_solution")
                )
            elif result.outcome == "verification_failure":
                return dict(
                    outcome="verification_failure", native=native, admitted=[], reason=result.reason
                )
        if verified:
            self.caches.put(key, native)
        return dict(
            outcome="verified_solution"
            if verified
            else native["outcome"]
            if native["outcome"] != "assignment_found"
            else "resource_exhausted",
            native=native,
            admitted=verified,
            metrics=self.caches.metrics(),
        )
