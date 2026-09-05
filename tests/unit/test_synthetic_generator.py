"""Unit tests for Synthetic Demonstration Generator across all algorithmic families."""

from oeis_learn.data.synthetic_generator import SyntheticDemonstrationGenerator


def test_generate_samples_all_families():
    gen = SyntheticDemonstrationGenerator(seed=42)
    families = [
        "POLYNOMIAL_LINEAR",
        "POLYNOMIAL_QUADRATIC",
        "POLYNOMIAL_CUBIC",
        "RECURRENCE_ORDER1",
        "RECURRENCE_FIBONACCI",
        "MODULAR_PERIODIC",
    ]

    for fam in families:
        sample = gen.generate_sample(1, family=fam)
        assert sample is not None, f"Failed generating sample for family {fam}"
        assert sample.family == fam
        assert len(sample.terms) == 20
        assert "(module" in sample.wat_code
        # Verify first 3 terms are integers
        assert all(isinstance(x, int) for x in sample.terms[:3])


def test_generate_dataset_batch():
    gen = SyntheticDemonstrationGenerator(seed=99)
    dataset = gen.generate_dataset(num_samples=10)

    assert dataset.total_samples == 10
    assert len(dataset.samples) == 10
    for s in dataset.samples:
        assert s.byte_size > 0
