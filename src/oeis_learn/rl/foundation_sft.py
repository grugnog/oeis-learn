"""Strict offline SFT using the existing encoder/decoder and admitted pool view.

No source data, teachers, hidden truth or implicit device selection. Foundation
smoke is FP32/eager with null scheduler/scaler and completed-update checkpoints.
"""

from __future__ import annotations

from copy import deepcopy
import platform
from pathlib import Path
import random
import uuid
import numpy as np
import torch
import torch.nn.functional as F

from oeis_learn.data.program_pool import load_pool, load_trainer_view
from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.evaluation.foundation_cohort import write_json
from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    atomic_write,
    compute_canonical_digest as digest,
    compute_file_hash,
    load_json,
)
from oeis_learn.experiments.config import load_config
from oeis_learn.experiments.profiles import profile_digests
from oeis_learn.rl.sft_trainer import teacher_forced_logits
from oeis_learn.tracking.foundation_metrics import metric, resource_metrics
from oeis_learn.tracking.training_checkpoint import CheckpointStore


def runtime_record(device, *, run=None):
    device = torch.device(device)
    if str(device) not in ("cpu", "cuda"):
        raise ValueError("explicit cpu or cuda device required")
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.version.hip):
        raise ValueError("HIP GPU unavailable; no CPU fallback")
    root = Path(__file__).parents[1]
    # Pin the actual encoder/feature/codec/controller implementations as well as
    # the trainer. Constructor equality alone cannot detect changed forward math.
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".wat")
    )
    record = dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        machine=platform.machine(),
        device=str(device),
        hip=torch.version.hip,
        device_name=torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else platform.processor(),
        deterministic_algorithms=True,
        float32_matmul_precision="highest",
        allow_tf32=False,
        cpu_threads=4,
        dtype="fp32",
        implementation={str(p.relative_to(root)): compute_file_hash(p) for p in files},
    )
    if run is not None:
        record["run_parameters"] = deepcopy(run)
    return record


def prepare_run(
    config,
    pool,
    output,
    *,
    device,
    diagnostic,
    seed=20260913,
    budget_ns=600_000_000_000,
    diagnostic_small_host=False,
):
    """Controller validates full admission archive before exposing only trainer data."""
    if not diagnostic:
        raise ValueError(
            "training promotion requires measured provenance/isolation gates; use explicit --diagnostic for the bounded smoke"
        )
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(budget_ns) is not int
        or not 0 < budget_ns <= 600_000_000_000
    ):
        raise ValueError("invalid seed or smoke budget (maximum ten minutes)")
    runtime = runtime_record(device)
    cfg = load_config(config)
    pool_manifest = load_json((Path(pool) / "manifest.json").read_bytes())
    admitted = load_pool(pool, expected_pool_id=pool_manifest["pool_id"])
    if len(admitted.examples) < 4 or len(admitted.examples) % 4:
        raise ValueError("strict smoke requires a positive multiple of four admitted examples")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    bundle = root / "checkpoints"
    bundle.mkdir()
    atomic_write(ArtifactPath(bundle, "config.yaml"), Path(config).read_bytes())
    cfg.persist_effective(bundle)
    pool_record = dict(
        kind="admitted_pool",
        pool_id=admitted.pool_id,
        view_id=admitted.view_id,
        sample_ids=[x["program_id"] for x in admitted.examples],
    )
    for name in ("manifest.json", "examples.json"):
        atomic_write(
            ArtifactPath(root / "trainer", name), (Path(pool) / "trainer" / name).read_bytes()
        )
    run_id = str(uuid.uuid4())
    effective_hash = digest(load_json((bundle / "effective-config.json").read_bytes()))
    contract = dict(
        schema_version="foundation/v1",
        kind="experiment_contract",
        purpose="diagnostic",
        run_id=run_id,
        profiles=profile_digests(),
        track="strict_generic",
        initialization="random",
        objective="sft",
        constitution_version="2.0.0",
        observed_terms=20,
        total_terms=100,
        index_policy="prefix_rebased_zero",
        enabled_subsystems=["sft"],
        run_provenance=dict(
            source_revision=None,
            dependency_lock=None,
            base_image_digest=None,
            final_image_id=None,
            environment_report=None,
            dataset_manifest=digest(pool_record),
            benchmark_manifest=None,
            evaluation_protocol=None,
            effective_config=effective_hash,
        ),
    )
    contract["contract_id"] = digest(contract)
    run = dict(
        schema_version="foundation/v1",
        run_id=run_id,
        device=device,
        diagnostic=True,
        seed=seed,
        budget_ns=budget_ns,
        pool_id=admitted.pool_id,
        view_id=admitted.view_id,
        contract_sha256=digest(contract),
        diagnostic_small_host=diagnostic_small_host,
        free_disk_floor_bytes=(1 if diagnostic_small_host else 100) << 30,
        host_available_floor_bytes=(256 << 20) if diagnostic_small_host else (16 << 30),
    )
    runtime["run_parameters"] = deepcopy(run)
    for name, record in (("contract", contract), ("pool", pool_record), ("runtime", runtime)):
        write_json(bundle, name + ".json", record)
    write_json(root, "run.json", run)
    return run


def per_program_loss(logits, targets):
    mask = targets != PAD_ID
    counts = mask.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("empty program target")
    losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD_ID, reduction="none")
    return ((losses * mask).sum(dim=1) / counts).mean()


class FoundationTrainer:
    def __init__(self, root, *, resume=False, store=None):
        self.root = Path(root)
        self.run = load_json((self.root / "run.json").read_bytes())
        self.bundle = self.root / "checkpoints"
        self.cfg = load_config(self.bundle / "config.yaml")
        self.device = torch.device(self.run["device"])
        expected_runtime = runtime_record(str(self.device), run=self.run)
        if load_json((self.bundle / "runtime.json").read_bytes()) != expected_runtime:
            raise ValueError("incompatible runtime")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        view = load_trainer_view(self.root / "trainer", expected_view_id=self.run["view_id"])
        self.examples = {x["program_id"]: x for x in view}
        pool_record = load_json((self.bundle / "pool.json").read_bytes())
        if (
            set(self.examples) != set(pool_record["sample_ids"])
            or pool_record["pool_id"] != self.run["pool_id"]
        ):
            raise ValueError("pool identity/example mismatch")
        self.store = store or CheckpointStore(
            self.bundle, free_floor_bytes=self.run["free_disk_floor_bytes"]
        )
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(0.8, (64 << 30) / total), self.device)
        self.named_rng = {
            name: torch.Generator().manual_seed(self.run["seed"] + i)
            for i, name in enumerate(("initialization", "data_order", "sampling"))
        }
        self.previous = None
        self.completed_update = 0
        if resume:
            path, loaded = self.store.recover(
                expected_runtime=expected_runtime, include_payload=True
            )
            if (
                loaded.manifest["run_id"] != self.run["run_id"]
                or loaded.manifest["contract_sha256"] != self.run["contract_sha256"]
            ):
                raise ValueError("checkpoint/run identity mismatch")
            self.encoder, self.decoder = (
                loaded.encoder.to(self.device),
                loaded.decoder.to(self.device),
            )
            self.completed_update = loaded.manifest["completed_update"]
            self.previous = compute_file_hash(path)
        else:
            if self.store.manifests():
                raise ValueError("existing checkpoint requires resume")
            random.seed(self.run["seed"])
            np.random.seed(self.run["seed"])
            torch.manual_seed(self.run["seed"])
            kwargs = self.cfg.model_constructor_kwargs()
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.named_rng["initialization"].get_state())
                self.encoder = TriStreamEncoder(**kwargs["encoder"])
                self.decoder = WatTransformerDecoder(**kwargs["decoder"])
                self.named_rng["initialization"].set_state(torch.get_rng_state())
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.order = dict(
                epoch=0, permutation=self._permutation(), cursor=0, accumulation_step=0
            )
            self._next()
        self.parameters = [*self.encoder.parameters(), *self.decoder.parameters()]
        training = self.cfg.effective_training_profile()
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            foreach=False,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        if resume:
            expected = {
                k: v
                for k, v in self.optimizer.state_dict()["param_groups"][0].items()
                if k != "params"
            }
            groups = loaded.payload["optimizer"]["param_groups"]
            if (
                len(groups) != 1
                or {k: v for k, v in groups[0].items() if k != "params"} != expected
            ):
                raise ValueError("optimizer settings differ from frozen training profile")
            self.optimizer.load_state_dict(loaded.payload["optimizer"])
            self.order = deepcopy(loaded.payload["data_order"])
            if len(self.order["permutation"]) != len(self.examples) or set(
                self.order["permutation"]
            ) != set(self.examples):
                raise ValueError("checkpoint permutation differs from admitted pool")
            self.restore_rng(loaded.payload)
        self.encoder.train()
        self.decoder.train()

    def _permutation(self):
        ids = list(self.examples)
        return [
            ids[i]
            for i in torch.randperm(len(ids), generator=self.named_rng["data_order"]).tolist()
        ]

    def _next(self):
        if self.order["cursor"] == len(self.order["permutation"]):
            self.order.update(
                epoch=self.order["epoch"] + 1, cursor=0, permutation=self._permutation()
            )
        cursor = self.order["cursor"]
        self.order["next_sample_ids"] = self.order["permutation"][cursor : cursor + 4]

    def peek(self):
        """Disposable prefetch cannot mutate order or any RNG stream."""
        return deepcopy([self.examples[i] for i in self.order["next_sample_ids"]])

    def update(self):
        if self.completed_update >= self.cfg.data["training"]["updates"]:
            raise ValueError("frozen update budget exhausted")
        batch = self.peek()
        tokens = [[BOS_ID] + x["body_tokens"] for x in batch]
        target = torch.full(
            (len(batch), max(map(len, tokens))), PAD_ID, dtype=torch.long, device=self.device
        )
        for i, row in enumerate(tokens):
            target[i, : len(row)] = torch.tensor(row, device=self.device)
        before = self.decoder.lm_head.weight.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        logits = teacher_forced_logits(
            self.encoder,
            self.decoder,
            [[int(x) for x in item["visible_terms"]] for item in batch],
            target,
            device=self.device,
            pad_id=PAD_ID,
        )
        loss = per_program_loss(logits, target[:, 1:])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if any(not bool(torch.isfinite(p).all()) for p in self.parameters):
            raise FloatingPointError("nonfinite model parameters")
        if torch.equal(before, self.decoder.lm_head.weight):
            raise RuntimeError("optimizer did not change output parameters")
        self.completed_update += 1
        sample_ids = list(self.order["next_sample_ids"])
        self.order["cursor"] += len(batch)
        self._next()
        return dict(
            kind="update",
            completed_update=self.completed_update,
            sample_ids=sample_ids,
            loss=metric(loss.item(), "nats", "per-program mean cross-entropy"),
            gradient_norm=metric(norm.item(), "norm", "clip_grad_norm preclip"),
            examples=metric(len(batch), "count", "admitted trainer batch"),
            tokens=metric(
                sum(len(x["body_tokens"]) for x in batch),
                "count",
                "nonpadding targets including EOS",
            ),
            parameters_changed=True,
            parameter_device=str(self.decoder.lm_head.weight.device),
            resources=resource_metrics(self.device),
        )

    def payload(self):
        numpy = np.random.get_state()
        return dict(
            model={"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()},
            optimizer=self.optimizer.state_dict(),
            scheduler=None,
            scaler=None,
            python_rng=random.getstate(),
            numpy_rng=dict(
                algorithm=numpy[0],
                keys=numpy[1].tolist(),
                position=numpy[2],
                has_gauss=numpy[3],
                cached_gaussian=numpy[4],
            ),
            torch_cpu_rng=torch.get_rng_state(),
            torch_gpu_rng={
                str(i): torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            }
            if self.device.type == "cuda"
            else {},
            named_rng={name: rng.get_state() for name, rng in self.named_rng.items()},
            data_order=deepcopy(self.order),
        )

    def restore_rng(self, payload):
        random.setstate(payload["python_rng"])
        n = payload["numpy_rng"]
        np.random.set_state(
            (
                n["algorithm"],
                np.array(n["keys"], dtype=np.uint32),
                n["position"],
                n["has_gauss"],
                n["cached_gaussian"],
            )
        )
        torch.set_rng_state(payload["torch_cpu_rng"])
        expected_gpu = (
            {str(i) for i in range(torch.cuda.device_count())}
            if self.device.type == "cuda"
            else set()
        )
        if set(payload["torch_gpu_rng"]) != expected_gpu or set(payload["named_rng"]) != set(
            self.named_rng
        ):
            raise ValueError("active RNG stream inventory mismatch")
        for key, state in payload["torch_gpu_rng"].items():
            torch.cuda.set_rng_state(state, int(key))
        for key, state in payload["named_rng"].items():
            self.named_rng[key].set_state(state)

    def save(self, ledger, *, final=False, hook=lambda _: None):
        if any(p.grad is not None for p in self.parameters):
            raise ValueError("checkpoint requires a completed update boundary")
        path = self.store.save(
            self.payload(),
            ledger=ledger,
            completed_update=self.completed_update,
            previous=self.previous,
            final=final,
            hook=hook,
        )
        self.previous = compute_file_hash(path)
        return path
