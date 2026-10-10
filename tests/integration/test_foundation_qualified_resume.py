from copy import deepcopy
import pytest
import torch
from torch import nn
from oeis_learn.curriculum.symple_bandit import Exp3SBanditScheduler
from oeis_learn.rl.policy import FrozenReference
from oeis_learn.tracking.training_checkpoint import (
    QualifiedCheckpointStore,
    capture_learning_state,
    restore_learning_state,
    learning_state_digest,
)
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


def test_active_reference_optimizer_curriculum_rng_and_next_update_resume(tmp_path):
    torch.manual_seed(7)
    encoder, decoder = nn.Linear(2, 2), nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    reference = FrozenReference(encoder, decoder)
    curriculum = Exp3SBanditScheduler(["a", "b", "c", "d", "e"], seed=7)

    def update():
        x = torch.randn(2, 2)
        optimizer.zero_grad()
        loss = decoder(encoder(x)).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach()), curriculum.sample_active_prompts()

    update()
    state = capture_learning_state(
        encoder,
        decoder,
        optimizer,
        sampler={"seed": 7, "cursor": 4},
        prefetch={"next": []},
        reference=reference,
        curriculum=curriculum,
    )
    identity = digest("test-contract")
    store = QualifiedCheckpointStore(tmp_path, free_floor_bytes=0)
    path = store.save_learning(
        state,
        contract_sha256=identity,
        profile_sha256=identity,
        runtime_sha256=identity,
        ledger_sequence=4,
        charged_budget_ns=123,
        completed_update=1,
    )
    expected = update()
    expected_parameters = deepcopy(decoder.state_dict())
    _, base, restored = store.load_learning(
        path, expected_contract=identity, expected_profile=identity, expected_runtime=identity
    )
    assert base["charged_budget_ns"] == 123
    record = store.load_learning(
        path, expected_contract=identity, expected_profile=identity, expected_runtime=identity
    )[0]["qualified"]
    assert record["modes"]["reference"] and record["modes"]["curriculum"]
    assert record["replay_state"] is record["search_state"] is record["scaler_sha256"] is None
    assert restore_learning_state(
        restored, encoder, decoder, optimizer, reference=reference, curriculum=curriculum
    ) == ({"seed": 7, "cursor": 4}, {"next": []})
    assert update() == expected
    assert learning_state_digest(decoder.state_dict()) == learning_state_digest(expected_parameters)
    # A torn newest publication must fall back without trusting latest.json.
    bad = store.save_learning(
        restored,
        contract_sha256=identity,
        profile_sha256=identity,
        runtime_sha256=identity,
        ledger_sequence=5,
        charged_budget_ns=200,
        completed_update=1,
    )
    bad.with_suffix(".pt").write_bytes(b"torn")
    recovered, _ = store.recover_learning(
        expected_contract=identity, expected_profile=identity, expected_runtime=identity
    )
    assert recovered == path
    with pytest.raises(ValueError, match="active"):
        restore_learning_state(restored, encoder, decoder, optimizer, curriculum=curriculum)


def test_replay_cursor_and_search_frontier_continue_with_same_next_update(tmp_path):
    from tests.unit.test_foundation_archive_curriculum import generic_fixture
    from oeis_learn.data.generic_programs import default_config
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.sandbox.runner import FoundationRunner
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE, PAD_ID, codec_digest
    from oeis_learn.rl.replay import FrozenReplay
    from oeis_learn.rl.sft_trainer import replay_update
    from oeis_learn.evaluation.search import SearchSession
    from tests.unit.test_foundation_search import Proposer, Runner

    sample, outputs, holdouts = generic_fixture()
    archive = ProgramArchive(tmp_path / "archive", holdouts=holdouts)
    with FoundationRunner(tmp_path / "runtime", backend="python_wasmtime") as runner:
        key = archive.admit(
            sample["canonical_source"],
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance={"config": default_config(), "sample": sample},
            runner=runner,
            request_id="resume-admission",
        )
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
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    replay = FrozenReplay(archive)
    search = SearchSession([1] * 20, portfolio="baseline16")
    search.step(Proposer(), Runner())
    replay_update(encoder, decoder, optimizer, replay, generic=[key] * 3)
    state = capture_learning_state(
        encoder,
        decoder,
        optimizer,
        sampler={"cursor": 1},
        prefetch={"next": [key] * 3},
        replay=replay,
        search=search,
    )
    expected = replay_update(encoder, decoder, optimizer, replay, generic=[key] * 3)
    parameters = learning_state_digest(decoder.state_dict())
    next_candidate = search.step(Proposer(), Runner())
    restore_learning_state(state, encoder, decoder, optimizer, replay=replay, search=search)
    actual = replay_update(encoder, decoder, optimizer, replay, generic=[key] * 3)
    # Replay guarantees the next mathematical update; host timings/resources are
    # fresh observations and must not be copied from the previous invocation.
    for field in ("loss", "examples", "replay_examples", "replay_fraction", "normalization", "entropy"):
        assert actual[field] == expected[field]
    assert actual["phase_timings_ns"].keys() == expected["phase_timings_ns"].keys()
    assert all(v >= 0 for v in actual["phase_timings_ns"].values())
    assert learning_state_digest(decoder.state_dict()) == parameters
    assert search.step(Proposer(), Runner()) == next_candidate
    assert replay.cursor == 2
