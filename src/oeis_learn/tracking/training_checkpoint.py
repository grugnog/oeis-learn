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


def learning_state_digest(value):
    """Stable identities for tensor/RNG/component state, independent of pickle IDs."""
    import numpy as np
    from oeis_learn.sandbox.pipeline import digest_bytes

    def portable(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {
                "tensor": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": digest_bytes(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()),
            }
        if isinstance(item, np.ndarray):
            return {
                "numpy": str(item.dtype),
                "shape": list(item.shape),
                "bytes": digest_bytes(item.tobytes()),
            }
        if isinstance(item, dict):
            return {
                "mapping": sorted(
                    [[portable(k), portable(v)] for k, v in item.items()],
                    key=lambda row: repr(row[0]),
                )
            }
        if isinstance(item, (tuple, list)):
            return {"tuple" if isinstance(item, tuple) else "list": [portable(v) for v in item]}
        if item is None or type(item) in (bool, str, int, float):
            return item
        raise ValueError("unsupported checkpoint state type")

    return digest(portable(value))


def capture_learning_state(
    encoder,
    decoder,
    optimizer,
    *,
    sampler,
    prefetch,
    reference=None,
    replay=None,
    self_training=None,
    curriculum=None,
    search=None,
    scaler=None,
    scheduler=None,
):
    import random
    import numpy as np

    components = {
        "replay": replay,
        "self_training": self_training,
        "curriculum": curriculum,
        "search": search,
        "scaler": scaler,
        "scheduler": scheduler,
    }
    from copy import deepcopy

    return deepcopy(
        {
            "version": "learning-state/v1",
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "module_modes": {
                name: {key: module.training for key, module in model.named_modules()}
                for name, model in (("encoder", encoder), ("decoder", decoder))
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "reference": (
                None
                if reference is None
                else {
                    "encoder": reference.encoder.state_dict(),
                    "decoder": reference.decoder.state_dict(),
                }
            ),
            **{key: None if obj is None else obj.state_dict() for key, obj in components.items()},
            "sampler": sampler,
            "prefetch": prefetch,
        }
    )


def restore_learning_state(
    state,
    encoder,
    decoder,
    optimizer,
    *,
    reference=None,
    replay=None,
    self_training=None,
    curriculum=None,
    search=None,
    scaler=None,
    scheduler=None,
):
    import random
    import numpy as np

    expected = {
        "version",
        "encoder",
        "decoder",
        "optimizer",
        "module_modes",
        "rng",
        "reference",
        "replay",
        "self_training",
        "curriculum",
        "search",
        "scaler",
        "scheduler",
        "sampler",
        "prefetch",
    }
    if set(state) != expected or state["version"] != "learning-state/v1":
        raise ValueError("incomplete learning state")
    objects = {
        "reference": reference,
        "replay": replay,
        "self_training": self_training,
        "curriculum": curriculum,
        "search": search,
        "scaler": scaler,
        "scheduler": scheduler,
    }
    if any((obj is None) != (state[key] is None) for key, obj in objects.items()):
        raise ValueError("active checkpoint modes differ")
    if len(state["rng"]["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA RNG topology differs")
    for name, model in (("encoder", encoder), ("decoder", decoder)):
        if set(state["module_modes"][name]) != dict(model.named_modules()).keys():
            raise ValueError("module topology differs")
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    optimizer.load_state_dict(state["optimizer"])
    if reference is not None:
        reference.encoder.load_state_dict(state["reference"]["encoder"])
        reference.decoder.load_state_dict(state["reference"]["decoder"])
        reference.encoder.eval().requires_grad_(False)
        reference.decoder.eval().requires_grad_(False)
        from oeis_learn.rl.policy import model_digest

        reference.encoder_sha256, reference.decoder_sha256 = (
            model_digest(reference.encoder),
            model_digest(reference.decoder),
        )
    for key, obj in objects.items():
        if key != "reference" and obj is not None:
            obj.load_state_dict(state[key])
    for name, model in (("encoder", encoder), ("decoder", decoder)):
        for key, module in model.named_modules():
            module.training = state["module_modes"][name][key]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["sampler"], state["prefetch"]


class QualifiedCheckpointStore(CheckpointStore):
    """Versioned learning-state publication, separate from foundation/v1 blobs.

    The immutable learning base manifest binds profile, ledger and every active
    payload. QualifiedCheckpoint supplies active/null mode summaries. Recovery
    verifies the entire blob before returning it and never refunds ledger time.
    """

    def save_learning(
        self,
        state,
        *,
        contract_sha256,
        profile_sha256,
        runtime_sha256,
        ledger_sequence,
        charged_budget_ns,
        completed_update,
        previous=None,
        hook=lambda _: None,
    ):
        from oeis_learn.experiments.qualified_models import seal_record
        from oeis_learn.experiments.artifacts import atomic_write, ArtifactPath, canonical_bytes
        from oeis_learn.experiments.models import check_digest

        for name, value in (
            ("contract", contract_sha256),
            ("profile", profile_sha256),
            ("runtime", runtime_sha256),
        ):
            check_digest(value, name)
        if (
            state["version"] != "learning-state/v1"
            or state["sampler"] is None
            or state["prefetch"] is None
        ):
            raise ValueError("complete named sampler/prefetch state required")
        if any(
            type(v) is not int or v < 0
            for v in (ledger_sequence, charged_budget_ns, completed_update)
        ):
            raise ValueError("committed counters required")
        import re

        generations = [
            int(match.group(1))
            for p in self.root.rglob("*")
            if (match := re.search(r"qualified-([0-9]{8})", p.name))
        ]
        generation = max(generations, default=-1) + 1
        name = f"qualified-{generation:08d}"
        temporary = self.root / (".pending-" + name)
        # Reserve one full generation plus recovery fallback before writing.
        import io

        stream = io.BytesIO()
        torch.save(state, stream)
        encoded = stream.getvalue()
        reserve = 2 * len(encoded) + (1 << 20)
        if (
            tree_bytes(self.root) + reserve > self.quota
            or self.disk_usage(self.root).free - reserve < self.free_floor
        ):
            raise OSError("qualified checkpoint quota/free-disk reserve exhausted")
        with temporary.open("xb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        hook("blob_flushed")
        blob = self.root / (name + ".pt")
        os.rename(temporary, blob)
        fsync_directory(self.root)
        hook("blob_published")
        identities = {
            key: None if value is None else learning_state_digest(value)
            for key, value in state.items()
            if key != "version"
        }
        base = {
            "version": "learning-base/v1",
            "blob_sha256": compute_file_hash(blob),
            "blob_path": blob.name,
            "profile_sha256": profile_sha256,
            "runtime_sha256": runtime_sha256,
            "completed_update": completed_update,
            "ledger_sequence": ledger_sequence,
            "charged_budget_ns": charged_budget_ns,
            "states": identities,
        }
        base_hash = digest(base)
        atomic_write(ArtifactPath(self.root, name + "-base.json"), canonical_bytes(base))
        modes = {
            key: state[key] is not None
            for key in (
                "reference",
                "replay",
                "self_training",
                "curriculum",
                "search",
                "scaler",
                "scheduler",
            )
        }
        replay, curriculum, search = state["replay"], state["curriculum"], state["search"]
        record = seal_record(
            "qualified_checkpoint",
            base_manifest_sha256=base_hash,
            contract_sha256=contract_sha256,
            modes=modes,
            reference_encoder_sha256=(
                None
                if state["reference"] is None
                else learning_state_digest(state["reference"]["encoder"])
            ),
            reference_decoder_sha256=(
                None
                if state["reference"] is None
                else learning_state_digest(state["reference"]["decoder"])
            ),
            replay_state=(
                None
                if replay is None
                else {
                    "archive_root_sha256": replay["snapshot"]["root_sha256"],
                    "cursor": replay["cursor"],
                }
            ),
            frozen_round=(
                None if state["self_training"] is None else state["self_training"]["round"]
            ),
            curriculum_state=(
                None
                if curriculum is None
                else {
                    "statistics_sha256": identities["curriculum"],
                    "rng_sha256": learning_state_digest(curriculum["rng"]),
                }
            ),
            search_state=(
                None
                if search is None
                else {
                    "frontier_sha256": learning_state_digest(search["frontier"]),
                    "cursor": search["cursor"],
                }
            ),
            sampler_sha256=identities["sampler"],
            prefetch_sha256=identities["prefetch"],
            scaler_sha256=identities["scaler"],
            scheduler_sha256=identities["scheduler"],
            stream_state=None,
            ledger_sequence=ledger_sequence,
            charged_budget_ns=charged_budget_ns,
            previous_sha256=previous,
        )
        manifest = {
            "version": "learning-checkpoint/v1",
            "base": name + "-base.json",
            "qualified": record,
        }
        from oeis_learn.experiments.artifacts import canonical_bytes

        atomic_write(ArtifactPath(self.root, name + ".json"), canonical_bytes(manifest))
        self.load_learning(
            self.root / (name + ".json"),
            expected_contract=contract_sha256,
            expected_profile=profile_sha256,
            expected_runtime=runtime_sha256,
        )
        hook("manifest_published")
        replace_json(
            self.root / "qualified-latest.json",
            {"path": name + ".json", "sha256": compute_file_hash(self.root / (name + ".json"))},
        )
        self.prune_learning()
        return self.root / (name + ".json")

    def load_learning(
        self, manifest_path, *, expected_contract, expected_profile, expected_runtime
    ):
        from oeis_learn.experiments.qualified_models import validate_record

        manifest_path = Path(manifest_path)
        if manifest_path.parent.resolve() != self.root.resolve():
            raise ValueError("checkpoint must belong to this store")
        manifest = load_json(manifest_path.read_bytes())
        if (
            set(manifest) != {"version", "base", "qualified"}
            or manifest["version"] != "learning-checkpoint/v1"
            or manifest["base"] != manifest_path.stem + "-base.json"
        ):
            raise ValueError("invalid learning checkpoint manifest")
        record = manifest["qualified"]
        validate_record(record)
        base = load_json((self.root / manifest["base"]).read_bytes())
        if (
            digest(base) != record["base_manifest_sha256"]
            or record["contract_sha256"] != expected_contract
            or base["profile_sha256"] != expected_profile
            or base["runtime_sha256"] != expected_runtime
            or base["blob_path"] != manifest_path.stem + ".pt"
        ):
            raise ValueError("checkpoint identity mismatch")
        blob = self.root / base["blob_path"]
        if compute_file_hash(blob) != base["blob_sha256"]:
            raise ValueError("checkpoint blob mismatch")
        state = torch.load(blob, map_location="cpu", weights_only=False)
        identities = {
            key: None if value is None else learning_state_digest(value)
            for key, value in state.items()
            if key != "version"
        }
        if (
            identities != base["states"]
            or record["ledger_sequence"] != base["ledger_sequence"]
            or record["charged_budget_ns"] != base["charged_budget_ns"]
        ):
            raise ValueError("incomplete/tampered checkpoint state")
        for key in record["modes"]:
            if record["modes"][key] != (state[key] is not None):
                raise ValueError("active mode summary mismatch")
        for name in ("sampler", "prefetch", "scaler", "scheduler"):
            if record[name + "_sha256"] != identities[name]:
                raise ValueError("component summary mismatch")
        if state["reference"] is not None and (
            record["reference_encoder_sha256"]
            != learning_state_digest(state["reference"]["encoder"])
            or record["reference_decoder_sha256"]
            != learning_state_digest(state["reference"]["decoder"])
        ):
            raise ValueError("reference summary mismatch")
        if state["replay"] is not None and record["replay_state"] != {
            "archive_root_sha256": state["replay"]["snapshot"]["root_sha256"],
            "cursor": state["replay"]["cursor"],
        }:
            raise ValueError("replay summary mismatch")
        if (
            state["self_training"] is not None
            and record["frozen_round"] != state["self_training"]["round"]
        ):
            raise ValueError("round summary mismatch")
        if state["curriculum"] is not None and record["curriculum_state"] != {
            "statistics_sha256": identities["curriculum"],
            "rng_sha256": learning_state_digest(state["curriculum"]["rng"]),
        }:
            raise ValueError("curriculum summary mismatch")
        if state["search"] is not None and record["search_state"] != {
            "frontier_sha256": learning_state_digest(state["search"]["frontier"]),
            "cursor": state["search"]["cursor"],
        }:
            raise ValueError("search summary mismatch")
        return manifest, base, state

    def prune_learning(self):
        manifests = [
            p
            for p in sorted(self.root.glob("qualified-[0-9]*.json"), reverse=True)
            if not p.stem.endswith("-base")
        ]
        # The writer has just consumer-validated the newest generation. Preserve
        # a complete fallback; corruption of a retained one fails before pruning.
        for path in manifests[:2]:
            manifest = load_json(path.read_bytes())
            base = load_json((self.root / manifest["base"]).read_bytes())
            if (
                digest(base) != manifest["qualified"]["base_manifest_sha256"]
                or compute_file_hash(self.root / base["blob_path"]) != base["blob_sha256"]
            ):
                raise ValueError("retained checkpoint is corrupt; pruning aborted")
        for path in manifests[2:]:
            (self.root / (path.stem + ".pt")).unlink()
            (self.root / (path.stem + "-base.json")).unlink()
            path.unlink()
        fsync_directory(self.root)

    def recover_learning(self, **identities):
        # Newest-invalid blobs cannot advertise readiness. Preserve older valid
        # generations, and leave the append-only budget ledger authoritative.
        failures = []
        for path in sorted(self.root.glob("qualified-[0-9]*.json"), reverse=True):
            if path.stem.endswith("-base"):
                continue
            try:
                return path, self.load_learning(path, **identities)
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                EOFError,
                pickle.UnpicklingError,
            ) as exc:
                # Runtime/contract drift is not corrupt storage. Leave valid
                # bytes untouched and require a compatible continuation.
                if isinstance(exc, ValueError) and str(exc) == "checkpoint identity mismatch":
                    raise
                failures.append(str(exc))
                for component in (
                    path.with_suffix(".pt"),
                    path.with_name(path.stem + "-base.json"),
                    path,
                ):
                    self._quarantine(component)
        raise ValueError("no complete qualified checkpoint remains: " + "; ".join(failures))
