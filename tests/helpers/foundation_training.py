"""Small real admitted-pool fixtures; no hand-written program teachers."""

from pathlib import Path
import pytest
import yaml
from tests.integration.test_foundation_bootstrap import setup_inputs
from oeis_learn.data.program_pool import build_pool
from oeis_learn.rl.foundation_sft import prepare_run, FoundationTrainer
from oeis_learn.tracking.training_checkpoint import CheckpointStore


@pytest.fixture(scope="session")
def training_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("training-inputs")
    sampler, cohort = setup_inputs(root / "source-inputs")
    pool = root / "pool"
    report = build_pool(sampler, cohort, pool)
    assert report["status"] == "complete", report
    raw = yaml.safe_load(Path("configs/foundation/wat_smoke.yaml").read_text())
    raw["model"].update(d_model=16, heads=2, encoder_layers=1, decoder_layers=1, feed_forward=32)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    return config, pool


def prepare(root, inputs):
    return prepare_run(*inputs, root, device="cpu", diagnostic=True)


def trainer(root, *, resume=False):
    return FoundationTrainer(
        root, resume=resume, store=CheckpointStore(root / "checkpoints", free_floor_bytes=0)
    )
