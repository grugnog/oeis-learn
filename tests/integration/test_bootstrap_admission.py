"""Real Wasmtime/reference integration; this is CPU evidence, not HIP acceptance."""

import pytest
from oeis_learn.data.bootstrap_programs import ConnectedSampler, default_config
from oeis_learn.data.structural_splits import StructuralHoldouts
from oeis_learn.decoder.program_codec import encode_body
from oeis_learn.discovery.program_archive import ProgramArchive
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.fresh.replay import admit, restore
from oeis_learn.sandbox.runner import FoundationRunner


def test_connected_admission_and_independent_replay(tmp_path):
    generator = ConnectedSampler(default_config())
    runner = FoundationRunner(tmp_path / "execution", backend="python_wasmtime")
    archive = ProgramArchive(tmp_path / "archive", holdouts=StructuralHoldouts())
    try:
        for stage in range(4):
            sample = generator.sample(stage * 16)
            evidence = runner.run_single(
                sample["canonical_source"], range(100), request_id=f"stage-{stage}"
            )
            if evidence.outcome == "numeric_limit":
                # Bounded state multipliers can overflow: admission must reject.
                continue
            assert evidence.outcome is None, evidence.reason
            values = list(map(int, evidence.outputs))
            key = archive.admit(
                sample["canonical_source"],
                values[:20],
                values,
                origin="generic_sample",
                provenance={"config": generator.config, "sample": sample},
                runner=runner,
                request_id=f"admit-{stage}",
            )
            assert archive.get(key)["source"] == sample["canonical_source"]

        # Select a seed for which both generic structures are training eligible.
        original = "i256.const 7 i256.const 0 i256.add"
        shorter = "i256.const 7"
        seed = next(
            s
            for s in range(100)
            if all(StructuralHoldouts(seed=s).decision(p, [7] * 20)[0] for p in (original, shorter))
        )
        replay_archive = ProgramArchive(tmp_path / "replay", holdouts=StructuralHoldouts(seed=seed))
        rows = [
            dict(program_id=digest("original"), values=["7"] * 20, tokens=encode_body(original))
        ]
        candidate = dict(
            program_id=rows[0]["program_id"],
            matched=True,
            complete=True,
            source=shorter,
            tokens=encode_body(shorter),
            model_provenance=dict(
                method="greedy", encoder_sha256=digest("encoder"), decoder_sha256=digest("decoder")
            ),
        )
        report = dict(splits={"training": {"attempts": [candidate]}})
        records = admit(report, rows, replay_archive, runner, [], seconds=30)
        assert len(records) == 1
        assert restore(records, rows, replay_archive)[rows[0]["program_id"]] == encode_body(shorter)
        with pytest.raises(ValueError, match="training target"):
            restore(records, [], replay_archive)
        # A claimed prefix fit is insufficient: continuation mismatch fails admission.
        candidate["source"] = "i256.const 8"
        candidate["tokens"] = encode_body(candidate["source"])
        with pytest.raises(ValueError, match="execution mismatch"):
            admit(report, rows, replay_archive, runner, [], seconds=30)
    finally:
        runner.close()
