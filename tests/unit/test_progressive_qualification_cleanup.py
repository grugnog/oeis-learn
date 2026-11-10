"""CPU lifetime regressions with explicit HIP-boundary doubles, no device evidence."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from oeis_learn.generalization.progressive.contracts import digest, load
from oeis_learn.pipeline_validation import progressive_qualification as qualification
from oeis_learn.pipeline_validation import progressive_training as training


@pytest.mark.parametrize("length", [4, 20])
@pytest.mark.parametrize("retained_bytes", [0, 512])
def test_case_releases_memory_and_workspaces_without_waiving_leaks(
    tmp_path, monkeypatch, fitting_client, length, retained_bytes
):
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.pipeline_validation.progressive_workstation_fixture import profile

    # Exercise real CPU updates, checkpoint restore, generation and native score.
    # Only the device boundary and its allocator counters are diagnostic doubles.
    memories = []
    state = dict(workspace=0, retained=0, clears=0)
    original_runner = training.PolicyUpdate

    def runner(*args, **kwargs):
        kwargs["device"] = "cpu"
        result = original_runner(*args, **kwargs)
        forward = result.encoder.forward_from_sequences

        def encode(*args, **kwargs):
            kwargs["device"] = "cpu"
            value = forward(*args, **kwargs)
            memories.append(weakref.ref(value))
            state["workspace"] = 32 << 20
            state["retained"] = retained_bytes
            return value

        result.encoder.forward_from_sequences = encode
        return result

    def allocated():
        return state["workspace"] + state["retained"] + sum(
            ref().numel() * ref().element_size() for ref in memories if ref() is not None
        )

    def clear_workspaces():
        state["workspace"] = 0
        state["clears"] += 1

    class CPUDeviceBoundary:
        _C = SimpleNamespace(_cuda_clearCublasWorkspaces=clear_workspaces)
        cuda = SimpleNamespace(
            memory_allocated=allocated,
            memory_reserved=allocated,
            memory_stats=lambda: {"allocated_bytes.all.current": allocated()},
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 64 << 20,
            get_rng_state_all=lambda: [],
            synchronize=lambda: None,
            empty_cache=lambda: None,  # Does not release live workspaces/tensors.
        )

        def __getattr__(self, name):
            return getattr(torch, name)

        def tensor(self, *args, **kwargs):
            kwargs["device"] = "cpu"
            return torch.tensor(*args, **kwargs)

    monkeypatch.setattr(source_snapshot, "source_snapshot", lambda: {})
    monkeypatch.setattr(qualification, "require_hip", CPUDeviceBoundary)
    monkeypatch.setattr(qualification, "hardware_facts", lambda: {"diagnostic": "CPU-only"})
    monkeypatch.setattr(training, "PolicyUpdate", runner)
    request = qualification.matrix(
        [profile()], source=digest({}), image=digest("diagnostic-image"), engine="podman",
        hardware={"diagnostic": "CPU-only"}, input_lengths=(length,), microbatches=(7,),
    )
    monkeypatch.setenv("OEIS_QUALIFICATION_IMAGE", request["image"])
    output = tmp_path / "case"
    if retained_bytes:
        with pytest.raises(ValueError, match="owned HIP tensors remain.*512 bytes"):
            qualification.run_case(request, request["cases"][0], output, client=fitting_client)
        assert not (output / "producer.json").exists()
    else:
        result = qualification.run_case(request, request["cases"][0], output, client=fitting_client)
        assert result["measurements"]["cleanup"]
    measures = load(output / "measurements.json")
    assert measures["full_restore"] and measures["generation"]
    assert measures["cleanup"] == (retained_bytes == 0)
    assert measures["cleanup_allocated_bytes"] == retained_bytes
    assert measures["cleanup_diagnostics"]["before_workspace_release_bytes"] == (32 << 20) + retained_bytes
    assert state["clears"] == 2  # Normalize baseline and clean up after execution.
    assert all(ref() is None for ref in memories)
