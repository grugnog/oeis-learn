"""Resource failures identify the actual run filesystem and retain frozen limits."""

from types import SimpleNamespace
import pytest
from oeis_learn.tracking import foundation_controller as controller


@pytest.mark.parametrize("free", [2 << 30, (100 << 30) - 1])
def test_free_space_error_reports_run_path_and_required_capacity(tmp_path, free):
    guard = controller.ResourceGuard(
        tmp_path, free_floor=100 << 30, disk_usage=lambda root: SimpleNamespace(free=free)
    )
    with pytest.raises(OSError) as error:
        guard.check()
    assert str(tmp_path) in str(error.value)
    assert f"free_bytes={free}" in str(error.value)
    assert f"minimum_free_bytes={100 << 30}" in str(error.value)


def test_exact_free_floor_passes_and_measures_run_root(tmp_path, monkeypatch):
    sampled = []
    resources = {"host_available": {"state": "measured", "value": 32 << 30}}

    def disk_usage(root):
        sampled.append(root)
        return SimpleNamespace(free=100 << 30)

    monkeypatch.setattr(controller, "resource_metrics", lambda device: resources)
    guard = controller.ResourceGuard(tmp_path, free_floor=100 << 30, disk_usage=disk_usage)
    assert guard.check() == resources and sampled == [tmp_path]
    assert guard.quota == 200 << 30 and guard.host_floor == 16 << 30


def test_run_quota_error_reports_usage_without_refunding_or_deleting_bytes(tmp_path):
    artifact = tmp_path / "checkpoint"
    artifact.write_bytes(b"1234567890")

    def unexpected_disk_usage(root):
        pytest.fail("quota must stop before querying free space")

    guard = controller.ResourceGuard(
        tmp_path, free_floor=100, quota=10, disk_usage=unexpected_disk_usage
    )
    with pytest.raises(OSError) as error:
        guard.check()
    assert str(tmp_path) in str(error.value)
    assert "used_bytes=10, quota_bytes=10" in str(error.value)
    assert artifact.read_bytes() == b"1234567890"
