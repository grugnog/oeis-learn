import pytest

from oeis_learn.pipeline_validation.progressive_velocity_worker import rehearse_batch


def profile():
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.models import resolve_profile

    base = resolve({"version": "bigint-bootstrap/v1"})
    return resolve_profile(base, shape="smaller_width", seed=29)


@pytest.mark.parametrize("batch_size", [16, 32])
def test_interrupted_backward_and_two_restores(tmp_path, batch_size):
    result = rehearse_batch(tmp_path, profile(), batch_size=batch_size, device="cpu")
    assert result["restored_weights"] == result["uninterrupted_weights"]
    assert result["restored_optimizer"] == result["uninterrupted_optimizer"]
    assert result["next_rows"] == result["uninterrupted_next_rows"]
    assert result["useful_updates"] == 2 and result["discarded_updates"] == 1
    assert (
        bool(result["matching_eligible"]) == bool(result["matching_passes"]) == (batch_size == 32)
    )
    assert result["charged_tokens"] == result["useful_tokens"] + result["discarded_tokens"]
    assert result["discarded_tokens"] > 0 and result["checkpoint_bytes"] > 0


@pytest.mark.gpu
def test_actual_hip_restore(tmp_path):
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires the workstation HIP device")
    for size in (16, 32):
        assert rehearse_batch(tmp_path / str(size), profile(), batch_size=size, device="cuda")[
            "full_restore"
        ]
