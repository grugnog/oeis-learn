"""Contract test for Synthetic Demonstration Dataset JSON Schema."""

import json
from oeis_learn.data.models import SyntheticDemonstrationPair
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationDataset, SyntheticDemonstrationGenerator


def test_sft_dataset_schema_and_serialization(tmp_path):
    gen = SyntheticDemonstrationGenerator(seed=123)
    sample = gen.generate_sample(0, family="POLYNOMIAL_QUADRATIC")

    assert sample is not None
    assert sample.sample_id.startswith("SYNTH_POLYNOMIAL_QUADRATIC_")
    assert len(sample.terms) == 20
    assert "(module" in sample.wat_code
    assert sample.byte_size > 0
    assert sample.lz_complexity > 0

    dataset = SyntheticDemonstrationDataset(
        version="1.0.0",
        total_samples=1,
        samples=[sample],
    )
    json_path = tmp_path / "test_sft.json"
    gen.save_dataset(dataset, str(json_path))

    loaded = gen.load_dataset(str(json_path))
    assert loaded.version == "1.0.0"
    assert loaded.total_samples == 1
    assert loaded.samples[0].sample_id == sample.sample_id
    assert loaded.samples[0].terms == sample.terms
