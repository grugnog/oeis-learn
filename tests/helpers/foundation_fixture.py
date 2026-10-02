"""Diagnostic test checkpoints using the real encoder/Transformer classes.

A hand-wired token-transition fixture exercises weight loading and boundaries;
its reported accuracy is not a training result. No fixture model is in runtime code.
"""

from pathlib import Path
import random
import uuid
import numpy as np
import torch
import yaml
from oeis_learn.decoder.program_codec import BOS_ID, encode_body
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.experiments.config import load_config
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, compute_file_hash
from oeis_learn.experiments.profiles import profile_digests
from oeis_learn.evaluation.foundation_cohort import write_json, freeze_cohort
from oeis_learn.evaluation.foundation_synthesis import default_protocol


def make_checkpoint(root, body="i256.zero"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(Path("configs/foundation/wat_smoke.yaml").read_text())
    raw["model"].update(d_model=16, heads=2, encoder_layers=1, decoder_layers=1, feed_forward=32)
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    cfg = load_config(root / "config.yaml")
    cfg.persist_effective(root)
    kwargs = cfg.model_constructor_kwargs()
    with torch.random.fork_rng():
        torch.manual_seed(11)
        encoder, decoder = (
            TriStreamEncoder(**kwargs["encoder"]),
            WatTransformerDecoder(**kwargs["decoder"]),
        )
    tokens = encode_body(body)
    assert len(set([BOS_ID] + tokens[:-1])) == len(tokens)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.final_norm.weight.fill_(1)
        decoder.lm_head.bias.fill_(-20)
        for index, (before, after) in enumerate(zip([BOS_ID] + tokens[:-1], tokens)):
            decoder.token_embedding.weight[before, index] = 1000
            decoder.lm_head.weight[after, index] = 20
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=3e-4)
    # A real completed optimizer update populates moment state without changing
    # the diagnostic transition logits (all gradients are explicitly zero).
    for parameter in [*encoder.parameters(), *decoder.parameters()]:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    run_id = str(uuid.uuid4())
    sample_id = digest({"fixture_sample": 1})
    pool = {"kind": "diagnostic_pool", "sample_ids": [sample_id]}
    from oeis_learn.experiments.artifacts import load_json

    effective_hash = digest(load_json((root / "effective-config.json").read_bytes()))
    contract = {
        "schema_version": "foundation/v1",
        "kind": "experiment_contract",
        "purpose": "diagnostic",
        "run_id": run_id,
        "profiles": profile_digests(),
        "track": "strict_generic",
        "initialization": "random",
        "objective": "sft",
        "constitution_version": "2.0.0",
        "observed_terms": 20,
        "total_terms": 100,
        "index_policy": "prefix_rebased_zero",
        "enabled_subsystems": ["sft"],
        "run_provenance": {
            "source_revision": None,
            "dependency_lock": None,
            "base_image_digest": None,
            "final_image_id": None,
            "environment_report": None,
            "dataset_manifest": digest(pool),
            "benchmark_manifest": None,
            "evaluation_protocol": None,
            "effective_config": effective_hash,
        },
    }
    contract["contract_id"] = digest(contract)
    runtime = {
        "python": "3.12",
        "torch": str(torch.__version__),
        "device": "cpu",
        "purpose": "diagnostic",
    }
    for name, obj in [("contract", contract), ("pool", pool), ("runtime", runtime)]:
        write_json(root, name + ".json", obj)
    from oeis_learn.experiments.artifacts import load_json

    fields = dict(
        run_id=run_id,
        generation=0,
        completed_update=1,
        contract_sha256=digest(contract),
        pool_sha256=digest(pool),
        codec_sha256=profile_digests()["codec"],
        runtime_sha256=digest(runtime),
        effective_config_sha256=digest(load_json((root / "effective-config.json").read_bytes())),
        ledger_sequence=1,
        charged_budget_ns=1,
    )
    numpy = np.random.RandomState(1).get_state()
    payload = {
        "model": {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "scaler": None,
        "python_rng": random.Random(1).getstate(),
        "numpy_rng": {
            "algorithm": numpy[0],
            "keys": numpy[1].tolist(),
            "position": numpy[2],
            "has_gauss": numpy[3],
            "cached_gaussian": numpy[4],
        },
        "torch_cpu_rng": torch.Generator().manual_seed(1).get_state(),
        "torch_gpu_rng": {},
        "named_rng": {"sampling": torch.Generator().manual_seed(2).get_state()},
        "data_order": {
            "epoch": 0,
            "permutation": [sample_id],
            "cursor": 0,
            "next_sample_ids": [sample_id],
            "accumulation_step": 0,
        },
        "counters": fields,
    }
    torch.save(payload, root / "model.pt")
    manifest = dict(
        kind="checkpoint_manifest",
        schema_version="foundation/v1",
        **fields,
        blob_path="model.pt",
        blob_sha256=compute_file_hash(root / "model.pt"),
        next_sample_ids=[sample_id],
        payload_state_keys=list(payload),
        checkpoint_state="complete",
        previous_manifest_sha256=None,
    )
    write_json(root, "checkpoint.json", manifest)
    write_json(
        root,
        "ledger.json",
        {"run_id": run_id, "sequence": 1, "charged_budget_ns": 1, "status": "completed"},
    )
    return root / "checkpoint.json"


def make_cohort(root, rows=None, *, dev_count=1, final_count=0):
    root = Path(root)
    source = root / "source"
    source.mkdir(parents=True)
    if rows is None:
        rows = [record("zero", [0] * 100)]
    write_json(source, "records.json", rows)
    cfg = root / "cohort.yaml"
    cfg.write_text(
        yaml.safe_dump(
            dict(
                schema_version="foundation/v1",
                profile="prefix20_total100_v1",
                seed=1,
                dev_count=dev_count,
                final_count=final_count,
            )
        )
    )
    freeze_cohort(source, cfg, root / "cohort")
    return root / "cohort"


def record(name, values, offset=0, metadata=None):
    return {
        "record_id": "fixture:" + name,
        "first_index": offset,
        "indices": list(range(offset, offset + len(values))),
        "values": [str(x) for x in values],
        "metadata": metadata or {},
    }


def protocol_file(root, **changes):
    protocol = dict(default_protocol(), attempts=2, max_body_tokens=32)
    protocol.update(changes)
    path = Path(root) / "protocol.json"
    write_json(path.parent, path.name, protocol)
    return path
