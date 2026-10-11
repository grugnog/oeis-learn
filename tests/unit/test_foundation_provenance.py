"""Rejected replay provenance cannot be laundered by rehashing outer envelopes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from oeis_learn.discovery.program_archive import ProgramArchive
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.evaluation.foundation_cohort import write_json
from oeis_learn.evaluation.measurement_inputs import load_assets
from oeis_learn.sandbox.pipeline import Runtime
from tests.unit.test_foundation_archive_curriculum import generic_fixture
from oeis_learn.data.generic_programs import default_config


@pytest.fixture
def admitted(tmp_path):
    sample, outputs, holdouts = generic_fixture()
    archive = ProgramArchive(tmp_path / "archive", holdouts=holdouts)
    runtime = Runtime()
    runner = SimpleNamespace(
        run_single=lambda source, indices, **kw: runtime.evaluate(source, indices)
    )
    provenance = dict(config=default_config(), sample=sample)
    key = archive.admit(
        sample["canonical_source"],
        outputs[:20],
        outputs,
        origin="generic_sample",
        provenance=provenance,
        runner=runner,
        request_id="valid",
    )
    return archive, runner, outputs, archive.get(key)


def mutate(provenance, field):
    p = deepcopy(provenance)
    if field == "config":
        p["config"]["seed"] += 1
    elif field == "origin":
        p["sample"]["origin"] = "named_family_teacher"
    elif field in ("sample_seed", "sample_counter"):
        p["sample"][field] += 1
    elif field == "statistics":
        p["sample"]["statistics"] = {}
    else:
        p["sample"]["body_tokens"] = [0]
    return p


@pytest.mark.parametrize(
    "field", ["origin", "config", "sample_seed", "sample_counter", "statistics", "tokens"]
)
def test_admission_reload_and_measurement_enforce_replay_rejections(admitted, tmp_path, field):
    archive, runner, outputs, packet = admitted
    bad = mutate(packet["provenance"], field)
    before = {p.name: p.read_bytes() for p in archive.root.iterdir()}
    entries = list(archive._entries)
    with pytest.raises(ValueError, match="generic sample rejected"):
        archive.admit(
            packet["source"],
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance=bad,
            runner=runner,
            request_id="forged",
        )
    assert before == {p.name: p.read_bytes() for p in archive.root.iterdir()}
    assert archive._entries == entries
    packet["provenance"] = bad
    packet["entry"]["sample_sha256"] = digest(bad)
    packet["entry"]["identity_sha256"] = digest(packet["entry"], "identity_sha256")
    key = digest(packet)
    with pytest.raises(ValueError, match="generic sample rejected"):
        archive.validate_packet(packet, key)
    row = dict(key=key, **packet)
    values = {
        "examples": dict(
            version="measurement-examples/v1",
            rows=[row] * 128,
            training_rows=[row],
            validation_rows=[],
            structures=1,
            holdouts={},
            snapshot=dict(entries=[key]),
        ),
        "development": {},
        "model_profile": {},
        "gate7": dict(runtime={}),
        "gate8": dict(runtime={}),
        "gate_evidence": [],
        "baseline": {},
        "baseline_checkpoint": {},
    }
    root = tmp_path / "measurement"
    assets = {
        name: write_json(root, "assets/" + name + ".json", value) for name, value in values.items()
    }
    assets["image"] = digest("image")
    manifest = dict(
        assets=assets,
        sources=[row["source"]] * 128,
        identities=dict(runtime=digest([assets["image"], {}, {}])),
    )
    with pytest.raises(ValueError, match="generic sample rejected"):
        load_assets(manifest, root, replay_gates=False)
