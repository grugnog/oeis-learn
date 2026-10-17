import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID
from oeis_learn.experiments.artifacts import compute_canonical_digest
from oeis_learn.fresh.worker import models
from oeis_learn.tracking.training_checkpoint import learning_state_digest

ROOT = Path(os.environ.get("OEIS_PROBE_OUTPUT", "tmp-hybrid-probe")).resolve()
ARCHIVE = Path(
    os.environ.get("OEIS_PROBE_ARCHIVE", "upload/bootstrap-step1-hybrid-control-001-weights.zip")
).resolve()


def load():
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        receipt = json.loads(z.read("bundle.json"))
        for name, ref in receipt["files"].items():
            b = z.read(name)
            assert (
                len(b) == ref["bytes"]
                and "sha256:" + hashlib.sha256(b).hexdigest() == ref["sha256"]
            ), name
        run = json.loads(z.read("run.json"))
        selected = json.loads(z.read("finalization.json"))["checkpoint"]
        manifest = json.loads(z.read(selected))
        base = json.loads(z.read(str(Path(selected).parent / manifest["base"])))
        assert compute_canonical_digest(base) == manifest["qualified"]["base_manifest_sha256"]
        assert (
            manifest["qualified"]["contract_sha256"]
            == run["settings"]["learner"]["contract_sha256"]
        )
        assert base["profile_sha256"] == run["settings"]["learner"]["model"]["identity_sha256"]
        blob_name = str(Path(selected).parent / base["blob_path"])
        blob = ROOT / "selected.pt"
        payload = z.read(blob_name)
        assert "sha256:" + hashlib.sha256(payload).hexdigest() == base["blob_sha256"]
        if not blob.exists():
            blob.write_bytes(payload)
        assert "sha256:" + hashlib.sha256(blob.read_bytes()).hexdigest() == base["blob_sha256"]
        rows = json.loads(z.read("learnability-examples.json"))["training"]
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(blob, map_location="cpu", weights_only=True)
    assert {
        k: None if v is None else learning_state_digest(v)
        for k, v in state.items()
        if k != "version"
    } == base["states"]
    settings = dict(run["settings"]["learner"], device="cpu", cpu_threads=2)
    e, d, _counts = models(settings)
    e.load_state_dict(state["encoder"])
    d.load_state_dict(state["decoder"])
    e.eval()
    d.eval()
    seqs = [list(map(int, r["values"])) for r in rows]
    toks = torch.full(
        (len(rows), max(len(r["tokens"]) for r in rows) + 1), PAD_ID, dtype=torch.long
    )
    for i, r in enumerate(rows):
        toks[i, : len(r["tokens"]) + 1] = torch.tensor([BOS_ID] + r["tokens"])
    return e, d, state, settings, rows, seqs, toks


def describe(logits, targets, rows):
    mask = targets != PAD_ID
    correct = (logits.argmax(-1) == targets) | ~mask
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_ID, reduction="none"
    ).reshape_as(targets)
    return {
        "teacher_argmax_exact": int(correct.all(-1).sum()),
        "token_loss": float(loss.sum() / mask.sum()),
        "failures": [
            {"offset": r["values"][0], "positions": (torch.where(~correct[i])[0] + 1).tolist()}
            for i, r in enumerate(rows)
            if not correct[i].all()
        ],
    }
