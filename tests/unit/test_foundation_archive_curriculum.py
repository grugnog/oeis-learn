from copy import deepcopy
import pytest
import torch
from oeis_learn.curriculum.symple_bandit import Exp3SBanditScheduler, AdaGGroupAllocator
from oeis_learn.data.generic_programs import GenericSampler, default_config
from oeis_learn.data.structural_splits import (
    StructuralHoldouts,
    structural_signature,
    split_for_structure,
)
from oeis_learn.discovery.program_archive import ProgramArchive
from oeis_learn.rl.replay import FrozenReplay
from oeis_learn.rl.sft_trainer import replay_update
from oeis_learn.sandbox.reference import execute_term
from oeis_learn.sandbox.wat_ast import parse
from oeis_learn.sandbox.runner import FoundationRunner


def generic_fixture():
    sampler = GenericSampler(default_config())
    for counter in range(1000):
        sample = sampler.sample(counter)
        if sample["sampling_failure"] is not None or len(sample["body_tokens"]) > 70:
            continue
        program = parse(sample["canonical_source"])
        results = [execute_term(program, i, step_limit=10000) for i in range(24)]
        if all(result.outcome is None for result in results):
            source = sample["canonical_source"]
            signature = structural_signature(source)
            seed = next(i for i in range(1000) if split_for_structure(signature, i) == "train")
            return (
                sample,
                [r.value for r in results],
                StructuralHoldouts(seed=seed, reserved_compositions=()),
            )
    raise AssertionError("no bounded valid generic fixture")


def test_archive_atomic_admission_long_retention_frozen_replay_and_gradients(tmp_path):
    sample, outputs, holdouts = generic_fixture()
    source = sample["canonical_source"]
    provenance = {"config": default_config(), "sample": sample}
    archive = ProgramArchive(tmp_path / "archive", holdouts=holdouts)
    with FoundationRunner(tmp_path / "runtime", backend="python_wasmtime") as runner:
        with pytest.raises(ValueError):
            archive.admit(
                source,
                outputs[:20],
                outputs[:20] + [outputs[20] + 1],
                origin="generic_sample",
                provenance=provenance,
                runner=runner,
                request_id="invalid",
            )
        assert archive._entries == [] and list(archive.root.glob("*.json")) == []
        key = archive.admit(
            source,
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance=provenance,
            runner=runner,
            request_id="valid",
        )
        value = archive.get(key)
        assert value["entry"]["outputs"] == list(map(str, outputs))
        replay = FrozenReplay(archive)
        old_view = replay.state_dict()
        archive.admit(
            source,
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance=provenance,
            runner=runner,
            request_id="second",
        )
        assert replay.state_dict() == old_view
        from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
        from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
        from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE, PAD_ID, codec_digest

        encoder = IntegerByteEncoder(16, 2, 1, 32)
        decoder = WatTransformerDecoder(
            vocab_size=FOUNDATION_VOCAB_SIZE,
            d_model=16,
            n_heads=2,
            n_decoder_layers=1,
            d_ff=32,
            dropout=0,
            pad_idx=PAD_ID,
            codec_sha256=codec_digest(),
        )
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()), lr=0.001
        )
        before = decoder.lm_head.weight.detach().clone()
        result = replay_update(encoder, decoder, optimizer, replay, generic=[key] * 3)
        assert (
            result["replay_fraction"] == 0.25
            and replay.cursor == 1
            and replay.committed_examples == 1
        )
        assert not torch.equal(before, decoder.lm_head.weight)
        cursor = replay.cursor
        with pytest.raises(ValueError):
            replay_update(encoder, decoder, optimizer, replay, generic=[])
        assert replay.cursor == cursor
        short = ProgramArchive(tmp_path / "short", holdouts=holdouts, context_limit=2)
        long_key = short.admit(
            source,
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance=provenance,
            runner=runner,
            request_id="long",
        )
        assert short.get(long_key)["source"] == source
        assert short.get(long_key)["entry"]["eligibility_reason"] == "over_context"
        assert short.training_view() == ()
        assert archive.reverify(runner)["root_sha256"] == archive.freeze()["root_sha256"]


def test_aggregate_visits_exact_conditional_probabilities_adaptive_caps_and_resume():
    ids = ["a", "b", "c", "d", "e"]
    scheduler = Exp3SBanditScheduler(ids)
    for step in range(4):
        scheduler.update_feedback(
            "a", 8, 16, current_step=step, elapsed_s=2, selection_probability=0.2
        )
    assert scheduler.get_competence("a") == 0.5 and scheduler.get_competence_slope("a") == 0
    scheduler.update_feedback("b", 0, 16, elapsed_s=100, selection_probability=0.2)
    chosen = scheduler.sample_active_prompts(4)
    for sid in chosen:
        selection = scheduler.last_selection[sid]
        expected = scheduler.get_selection_probabilities(selection["remaining"])[sid]
        assert selection["probability"] == pytest.approx(expected)
    allocations = AdaGGroupAllocator().compute_group_sizes(ids[:4], scheduler)
    assert sum(allocations.values()) == 32 and all(2 <= n <= 16 for n in allocations.values())
    assert len(set(allocations.values())) > 1
    clone = Exp3SBanditScheduler(ids, seed=0)
    clone.load_state_dict(deepcopy(scheduler.state_dict()))
    assert clone.sample_active_prompts() == scheduler.sample_active_prompts()
    allocator = AdaGGroupAllocator()
    assert allocator.compute_group_sizes(["a"], scheduler) == {"a": 16}
    assert allocator.unallocated == 16
    with pytest.raises(ValueError, match="cost"):
        scheduler.update_feedback("a", 8, 16)


def test_search_hypotheses_are_sft_only_and_every_admission_keeps_holdouts(tmp_path):
    from oeis_learn.evaluation.search import SearchSession
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    sample, outputs, holdouts = generic_fixture()
    source = sample["canonical_source"]
    archive = ProgramArchive(tmp_path / "archive", holdouts=holdouts)

    class Proposer:
        def stochastic(self, visible, seed, ledger, **kwargs):
            ledger.charge(expansions=1, tokens=len(sample["body_tokens"]))
            return source, {
                "method": "greedy",
                "encoder_sha256": digest("encoder"),
                "decoder_sha256": digest("decoder"),
            }

    with FoundationRunner(tmp_path / "runtime", backend="python_wasmtime") as runner:
        session = SearchSession(outputs[:20], portfolio="greedy")
        report = session.run(Proposer(), runner)
        candidate = report["results"][0]
        packet = candidate["provenance"]["hypothesis"]
        model = digest([packet["encoder_sha256"], packet["decoder_sha256"]])
        key = archive.admit(
            source,
            outputs[:20],
            outputs,
            origin="training_prefix_hypothesis",
            provenance={"search": packet},
            model_sha256=model,
            runner=runner,
            request_id="search-admission",
        )
        assert archive.get(key)["entry"]["training_eligible"]
        # Same proven program is retained but excluded when its structural group
        # is reserved. A training-prefix origin does not waive the guard.
        excluded = ProgramArchive(
            tmp_path / "excluded",
            holdouts=StructuralHoldouts(
                seed=holdouts.seed, reserved_structures=frozenset({structural_signature(source)})
            ),
        )
        blocked = excluded.admit(
            source,
            outputs[:20],
            outputs,
            origin="training_prefix_hypothesis",
            provenance={"search": packet},
            model_sha256=model,
            runner=runner,
            request_id="excluded-admission",
        )
        assert excluded.get(blocked)["entry"]["eligibility_reason"] == "structural_holdout"
        assert excluded.training_view() == ()
        changed = deepcopy(packet)
        changed["visible_prompt_sha256"] = digest(["999"] * 20)
        changed["identity_sha256"] = digest(changed, "identity_sha256")
        before = list(archive._entries)
        with pytest.raises(ValueError, match="binding"):
            archive.admit(
                source,
                outputs[:20],
                outputs,
                origin="training_prefix_hypothesis",
                provenance={"search": changed},
                model_sha256=model,
                runner=runner,
                request_id="bad-search-admission",
            )
        assert archive._entries == before
