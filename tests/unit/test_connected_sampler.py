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
