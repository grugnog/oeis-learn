import time
from types import SimpleNamespace

import pytest
import torch

from oeis_learn.decoder.bigint_generation import GenerationRequest, sample_batch


def test_expired_shared_deadline_never_starts_sampling(monkeypatch):
    from oeis_learn.decoder.sampler import WatProgramSampler

    monkeypatch.setattr(
        WatProgramSampler, "sample_trajectories", lambda *a, **kw: pytest.fail("expired")
    )
    # Use a real tiny decoder so constructing the sampler still checks its codec.
    from oeis_learn.decoder.bigint_codec import FOUNDATION_VOCAB_SIZE, codec_digest
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

    decoder = WatTransformerDecoder(
        vocab_size=FOUNDATION_VOCAB_SIZE,
        codec_sha256=codec_digest(),
        d_model=8,
        n_heads=2,
        n_decoder_layers=1,
        d_ff=16,
    ).eval()
    requests = [
        GenerationRequest(torch.zeros(1, 2, 8), i, 0, time.monotonic() - 1) for i in range(2)
    ]
    results = sample_batch(decoder, requests)
    assert len(results) == 2
    assert all(
        r["tokens"] == [] and not r["complete"] and r["failure"] == "deadline" for r in results
    )


def test_timeout_preserves_partial_rows_without_fitting_or_repair(monkeypatch):
    from oeis_learn.decoder import sampler
    import oeis_learn.decoder.bigint_generation as generation

    class Interrupted:
        def __init__(self, *a, **kw):
            self.active_token_rows = [[0, 11, 12], [0, 21]]

        def sample_trajectories(self, memory, seeds, **kw):
            assert seeds == [17, 19]
            raise TimeoutError()

    monkeypatch.setattr(sampler, "WatProgramSampler", Interrupted)
    deadline = time.monotonic() + 60
    results = generation.sample_batch(
        None, [GenerationRequest(torch.zeros(1, 2, 8), seed, 0.8, deadline) for seed in (17, 19)]
    )
    assert [r["tokens"] for r in results] == [[11, 12], [21]]
    assert all(not r["complete"] and r["source"] == "" for r in results)


def test_portfolio_scheduler_keeps_attempt_seeds_and_serial_fit_order(monkeypatch):
    from oeis_learn.fresh.bigint_diagnostics import _drive_portfolios
    import oeis_learn.decoder.bigint_generation as generation

    events = []
    memory = torch.zeros(1, 2, 8)
    deadline = time.monotonic() + 1

    def portfolio(index):
        for position in range(8):
            reply = yield GenerationRequest(
                memory, index * 100 + position, 0 if position == 0 else 0.8, deadline
            )
            events.append(("fit", index, position, reply))
        return index

    def sample(decoder, requests):
        events.append(("sample", [r.seed for r in requests], [r.deadline for r in requests]))
        return [r.seed for r in requests]

    monkeypatch.setattr(generation, "sample_batch", sample)
    assert _drive_portfolios(None, [portfolio(i) for i in range(3)]) == [0, 1, 2]
    samples = [e for e in events if e[0] == "sample"]
    assert [e[1] for e in samples] == [[i * 100 + p for i in range(3)] for p in range(8)]
    assert all(e[2] == [deadline] * 3 for e in samples)
    assert [e[1:3] for e in events if e[0] == "fit"] == [(i, p) for p in range(8) for i in range(3)]


def test_incomplete_sampler_row_mapping_is_rejected(monkeypatch):
    from oeis_learn.decoder import sampler

    monkeypatch.setattr(
        sampler,
        "WatProgramSampler",
        lambda *a, **kw: SimpleNamespace(sample_trajectories=lambda *a, **kw: []),
    )
    with pytest.raises(RuntimeError, match="row mapping"):
        sample_batch(None, [GenerationRequest(torch.zeros(1, 2, 8), 17, 0, time.monotonic() + 1)])


@pytest.mark.parametrize('fit_outcome,fit_complete', [('unsat_in_scope', True), ('unknown', False), ('resource_exhausted', False)])
def test_production_controls_preserve_prompt_and_attempt_denominators(monkeypatch, fit_outcome, fit_complete):
    from oeis_learn.fresh.bigint_diagnostics import generate
    from oeis_learn.fresh.bigint_config import defaults
    from oeis_learn.decoder import sampler, bigint_codec as codec

    source = codec.decode_body(codec.encode_body("(program (return $index))"))
    tokens = (codec.BOS_ID, *codec.encode_body(source))
    calls = []

    class Sampler:
        def __init__(self, *a, **kw):
            self.temperature = kw["temperature"]

        def sample_trajectories(self, memory, seeds, **kw):
            calls.append((self.temperature, list(seeds), len(memory)))
            return [
                SimpleNamespace(tokens=tokens, source=source, complete=True, failure=None)
                for _ in seeds
            ]

    class Encoder(torch.nn.Linear):
        def forward_from_sequences(self, rows, device=None):
            return torch.zeros(len(rows), 20, 2)

    class Client:
        def execute(self, *a, **kw):
            return {"outcome": "failure"}

        def fit(self, *a, **kw):
            return {"outcome": fit_outcome, "admitted": [], "complete": fit_complete}

    monkeypatch.setattr(sampler, "WatProgramSampler", Sampler)
    rows = [
        dict(id=str(i), indices=list(map(str, range(20))), values=[str(i)] * 20) for i in range(10)
    ]
    result = generate(
        Encoder(2, 2),
        torch.nn.Linear(2, 2),
        rows,
        defaults(),
        Client(),
        deadline=time.monotonic() + 120,
    )
    assert set(result["records"]) == {r["id"] for r in rows}
    for record in result["records"].values():
        assert record["complete"] is fit_complete
        assert set(record["details"]) == {"correct", "shuffled", "absent", "template"}
        assert all(len(arm["details"]) == 8 for arm in record["details"].values())
    assert all(size in (8, 2) for _, _, size in calls)
    assert sum(size for _, _, size in calls) == 10 * 8 * 3
    correct = result["records"]["0"]["details"]["correct"]["details"]
    assert correct[0]["tokens"] == list(tokens[1:])


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_finished_row_and_token_capped_row_keep_independent_dispositions(temperature):
    from oeis_learn.decoder import bigint_codec as codec

    bodies = [
        codec.encode_body("(program (return $index))"),
        codec.encode_body("(program (return (add (int 1234) (int 5678))))"),
    ]

    class Scripted(torch.nn.Module):
        codec_sha256 = codec.codec_digest()

        def logits(self, cache):
            logits = torch.full((len(cache.ids), 1, codec.FOUNDATION_VOCAB_SIZE), -100.0)
            for i, target in enumerate(cache.ids):
                body = bodies[target]
                logits[i, 0, body[min(cache.position, len(body) - 1)]] = 100
            return logits

        def prefill(self, tokens, memory, capacity):
            cache = SimpleNamespace(ids=[int(x) for x in memory[:, 0, 0]], position=0)
            return self.logits(cache), cache

        def decode_step(self, tokens, cache):
            cache.position += 1
            return self.logits(cache)

    decoder = Scripted().eval()
    requests = [
        GenerationRequest(
            torch.tensor([[[float(i)]]]),
            17 + i,
            temperature,
            time.monotonic() + 60,
            max_length=len(bodies[1]) - 1,
        )
        for i in range(2)
    ]
    together = sample_batch(decoder, requests)
    alone = [sample_batch(decoder, [request])[0] for request in requests]
    for batch, single in zip(together, alone):
        assert batch["tokens"] == single["tokens"]
        assert batch["complete"] == single["complete"]
        assert batch["failure"] == single["failure"]
    assert together[0]["complete"] and together[0]["tokens"] == bodies[0]
    assert not together[1]["complete"] and together[1]["source"] == ""
    assert together[1]["failure"] == "token_cap_without_eos"
