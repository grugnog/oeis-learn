"""Complete foundation checkpoint publication and newest-valid recovery.

Sidecars are immutable for the run. The blob is durable before its manifest;
latest.json is only a hint, never the recovery authority.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
import shutil
import uuid
import torch

from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
from oeis_learn.evaluation.foundation_cohort import write_json
from oeis_learn.experiments.artifacts import (
    compute_file_hash,
    compute_canonical_digest as digest,
    load_json,
)
from oeis_learn.experiments.models import validate_artifact
from oeis_learn.experiments.profiles import profile_digests
from oeis_learn.tracking.foundation_metrics import fsync_directory, replace_json

SIDECARS = {
    "contract": "contract_sha256",
    "pool": "pool_sha256",
    "runtime": "runtime_sha256",
    "effective-config": "effective_config_sha256",
}


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


class CheckpointStore:
    def __init__(
        self,
        root,
        *,
        quota_bytes=40 << 30,
        free_floor_bytes=100 << 30,
        disk_usage=shutil.disk_usage,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not 0 < quota_bytes <= 40 << 30 or free_floor_bytes < 0:
            raise ValueError("invalid checkpoint quota")
        self.quota, self.free_floor, self.disk_usage = quota_bytes, free_floor_bytes, disk_usage

    def manifests(self):
        return sorted(self.root.glob("checkpoint-[0-9]*.json"), reverse=True)

    def next_generation(self):
        # Quarantined generations are never reused after rollback.
        numbers = [
            int(p.name.split("-")[1].split(".")[0])
            for p in self.root.rglob("checkpoint-[0-9]*.json")
        ]
        return max(numbers, default=-1) + 1

    def _quarantine(self, path):
        if path.exists():
            target = self.root / "quarantine" / uuid.uuid4().hex
            target.mkdir(parents=True)
            os.rename(path, target / path.name)
            fsync_directory(target)
            fsync_directory(self.root)

    def recover(self, *, expected_runtime=None, include_payload=False):
        for path in self.root.glob(".pending-*"):
            self._quarantine(path)
        valid_blobs = {p.with_suffix(".pt").name for p in self.manifests()}
        for blob in self.root.glob("checkpoint-*.pt"):
            if blob.name not in valid_blobs:
                self._quarantine(blob)
        failures = []
        for path in self.manifests():
            try:
                checkpoint = load_foundation_checkpoint(path, include_payload=include_payload)
            except (
                ValueError,
                OSError,
                RuntimeError,
                EOFError,
                KeyError,
                TypeError,
                pickle.UnpicklingError,
            ) as exc:
                failures.append(dict(path=path.name, reason=str(exc)))
                self._quarantine(path.with_suffix(".pt"))
                self._quarantine(path)
                continue
            if expected_runtime is not None and checkpoint.manifest["runtime_sha256"] != digest(
                expected_runtime
            ):
                raise ValueError("incompatible runtime; continuation cannot change device/software")
            replace_json(
                self.root / "latest.json", dict(path=path.name, sha256=compute_file_hash(path))
            )
            if failures:
                write_json(
                    self.root,
                    f"recovery-{uuid.uuid4().hex}.json",
                    dict(selected=path.name, rejected=failures),
                )
            return path, checkpoint
        raise ValueError("no complete valid checkpoint remains")

    def save(
        self, payload, *, ledger, completed_update, previous=None, final=False, hook=lambda _: None
    ):
        generation = self.next_generation()
        fields = dict(
            run_id=ledger["run_id"],
            generation=generation,
            completed_update=completed_update,
            codec_sha256=profile_digests()["codec"],
            ledger_sequence=ledger["sequence"],
            charged_budget_ns=ledger["charged_budget_ns"],
        )
        for name, field in SIDECARS.items():
            fields[field] = digest(load_json((self.root / (name + ".json")).read_bytes()))
        payload = dict(payload, counters=fields)
        name = f"checkpoint-{generation:08d}"
        temporary, blob, manifest_path = (
            self.root / (".pending-" + name),
            self.root / (name + ".pt"),
            self.root / (name + ".json"),
        )

        def size(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(size(v) for v in value)
            return 64

        largest = max(
            [size(payload) + (1 << 20)] + [p.stat().st_size for p in self.root.glob("*.pt")]
        )
        if (
            tree_bytes(self.root) + 2 * largest > self.quota
            or self.disk_usage(self.root).free - 2 * largest < self.free_floor
        ):
            raise OSError("checkpoint quota/free-disk reserve exhausted")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        hook("blob_flushed")
        blob_hash = compute_file_hash(temporary)
        if tree_bytes(self.root) > self.quota:
            raise OSError("checkpoint quota exhausted")
        os.rename(temporary, blob)
        fsync_directory(self.root)
        hook("blob_published")
        manifest = dict(
            schema_version="foundation/v1",
            kind="checkpoint_manifest",
            **fields,
            blob_path=blob.name,
            blob_sha256=blob_hash,
            next_sample_ids=payload["data_order"]["next_sample_ids"],
            payload_state_keys=list(payload),
            checkpoint_state="complete",
            previous_manifest_sha256=previous,
        )
        validate_artifact(manifest)
        write_json(self.root, manifest_path.name, manifest)
        # Use the same consumer validation as evaluation before advertising latest.
        load_foundation_checkpoint(manifest_path)
        hook("manifest_published")
        replace_json(
            self.root / "latest.json",
            dict(path=manifest_path.name, sha256=compute_file_hash(manifest_path)),
        )
        if final:
            replace_json(
                self.root / "final.json",
                dict(path=manifest_path.name, sha256=compute_file_hash(manifest_path)),
            )
        self.prune()
        return manifest_path

    def prune(self):
        manifests = self.manifests()
        keep = set(manifests[:2])
        pin = self.root / "final.json"
        if pin.exists():
            record = load_json(pin.read_bytes())
            pinned = self.root / record["path"]
            if pinned not in manifests or compute_file_hash(pinned) != record["sha256"]:
                raise ValueError("invalid final checkpoint pin")
            load_foundation_checkpoint(pinned)
            keep.add(pinned)
        # Verify retained generations before deleting any fallback.
        for path in keep:
            load_foundation_checkpoint(path)
        for path in manifests:
            if path not in keep:
                path.with_suffix(".pt").unlink()
                path.unlink()
        fsync_directory(self.root)
