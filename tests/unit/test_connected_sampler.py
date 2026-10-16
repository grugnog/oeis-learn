from oeis_learn.data.bootstrap_programs import ConnectedSampler, default_config
from oeis_learn.data.program_admission import validate_sample
from oeis_learn.data.structural_splits import structural_signature
from oeis_learn.data.sampler_factory import sampler


def test_connected_draws_replay_and_parameters_preserve_structure():
    generator = ConnectedSampler(default_config())
    lengths = []
    for structure in range(20):
        draws = [generator.sample(structure * 16 + i) for i in range(3)]
        assert len({structural_signature(d["canonical_source"]) for d in draws}) == 1
        for draw in draws:
            assert validate_sample(draw, generator) is None
            assert draw == sampler(default_config()).sample(draw["sample_counter"])
            lengths.append(len(draw["body_tokens"]))
    assert max(lengths) < 512


def test_old_sampler_identity_untouched():
    from oeis_learn.data.generic_programs import GenericSampler, default_config as old_config

    assert sampler(old_config()).sample(19) == GenericSampler(old_config()).sample(19)


def test_connected_programs_execute_and_stage_export_preserves_coverage():
    from oeis_learn.sandbox.wat_ast import parse
    from oeis_learn.sandbox.reference import execute_term
    from oeis_learn.fresh.input_export import validation_view

    generator = ConnectedSampler(default_config())
    rows = {}
    for n in range(0, 256, 16):
        sample = generator.sample(n)
        for index in (0, 5, 19):
            result = execute_term(parse(sample["canonical_source"]), index)
            assert result.outcome is None
        rows[str(n)] = dict(program_id=str(n), stage=sample["statistics"]["stage"])
    assert {r["stage"] for r in validation_view(rows, 4)} == {0, 1, 2, 3}


def test_stream_includes_nonlinear_composition_and_coupled_state():
    from oeis_learn.sandbox.reference import execute_term
    from oeis_learn.sandbox.wat_ast import parse

    sampler = ConnectedSampler(default_config())
    nonlinear, coupled, iteration_scalar = False, False, False
    for counter in range(0, 2048, 16):
        draw = sampler.sample(counter)
        source = draw["canonical_source"]
        values = [execute_term(parse(source), n) for n in range(5)]
        if draw["statistics"]["stage"] == 1 and all(v.outcome is None for v in values):
            second_differences = [
                values[i + 2].value - 2 * values[i + 1].value + values[i].value for i in range(3)
            ]
            nonlinear |= any(second_differences)
        coupled |= "local.get $v4" in source
        iteration_scalar |= "local.get $c0 i32.const 1 i32.add i64.extend_i32_u" in source
    assert nonlinear and coupled and iteration_scalar


def test_curriculum_places_carried_state_before_conditional_composition():
    generator = ConnectedSampler(default_config())
    assert "loop" in generator.sample(2 * 16)["canonical_source"]
    assert "( if " in generator.sample(3 * 16)["canonical_source"]


def test_connected_training_mixture_caps_behaviors_and_balances_stages():
    from collections import Counter
    from oeis_learn.fresh.bootstrap_views import training_mixture, behavior

    rows = {}
    for stage in range(4):
        for i in range(100):
            values = [str(i)] * 20 if i < 50 else [str(i * n + stage) for n in range(20)]
            key = f"{stage}-{i}"
            rows[key] = dict(program_id=key, values=values, stage=stage, tokens=[1] * (i + 1))
    result = training_mixture(rows, 128)
    assert len(result) == 128
    assert Counter(r["stage"] for r in result.values()) == {s: 32 for s in range(4)}
    assert sum(behavior(r) == "prefix_constant" for r in result.values()) == 6
    assert training_mixture(dict(reversed(list(rows.items()))), 128) == result
