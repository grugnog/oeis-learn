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



def test_registered_operation_recovers_receipt_before_commit_and_protects_holdback(tmp_path):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import run_operation

    grant = dict(identity=digest("operator"), resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=900_000_000,
                                                              attempts=3), holdback_us=300_000_000)
    journal = BudgetJournal(tmp_path / "accounting", grant["identity"], grant["resources"])
    coordinates = dict(study=grant["identity"], stage="qualification", method="fixed", seed=0,
                       look=0, panel="exposed", profile="common", input_mode="true", role="search", ordinal=0)
    request = seal(dict(operation="search", candidate=digest("candidate")))
    issued = []
    def execute(out, budget_us):
        assert 0 < budget_us <= 600_000_000
        issued.append(out)
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        if len(issued) == 1:
            raise ValueError("original fit stopped")
        atomic_write(out / "producer.json", seal(dict(request=request["identity"], result="checked")))
    def verify(out):
        producer = load(out / "producer.json")
        assert producer["request"] == request["identity"]
        return producer
    args = dict(area=tmp_path, journal=journal, grant=grant, coordinates=coordinates, request=request,
                cost={"attempts": 1}, execute=execute, verify=verify)
    with pytest.raises(ValueError, match="original fit stopped"):
        run_operation(**args)
    good = run_operation(**args)
    assert good["attempt"] == 1 and len(issued) == 2
    block = issued[-1].parent
    (block / "receipt.json").unlink()
    for _ in range(2):
        assert run_operation(**args) == good
    assert len(issued) == 2 and journal.snapshot()["consumed"]["attempts"] == 2
    assert load(block / "first-error.json")["reason"] == "original fit stopped"
    with pytest.raises(ValueError):
        run_operation(**dict(args, request=seal(dict(operation="different"))))
    other = dict(coordinates, ordinal=1)
    with pytest.raises(ValueError, match="holdback"):
        run_operation(**dict(args, coordinates=other, grant=dict(grant, holdback_us=900_000_000)))
    assert journal.snapshot()["remaining"]["elapsed_us"] >= grant["holdback_us"]


def test_donated_adamw_state_matches_one_real_next_update(tmp_path, monkeypatch):
    import os
    import shutil
    from oeis_learn.generalization.progressive import training_rows
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
    from oeis_learn.pipeline_validation.progressive_pilot_worker import train
    from oeis_learn.pipeline_validation.progressive_velocity_worker import _disposable_rows
    from oeis_learn.pipeline_validation.progressive_velocity_continuation import compare_next_update

    rows, _ = _disposable_rows()
    stream = seal(dict(order=sorted(rows), registration=digest('original'), pair=0,
                       seeds=dict(initialization=29), pairs=[dict(positive_row=sorted(rows)[0], label=None)]))
    package = seal(dict(version='round028-pilot-training/v1', profile=profile(), microbatch=8, stream=stream,
                        rows=[dict(row, draw_identity=key) for key, row in rows.items()]))
    monkeypatch.setattr(training_rows, 'GENERATOR_LOOKS', (0, 12, 4_000_000))
    path = tmp_path / 'training.json'
    atomic_write(path, package)
    data = dict(training=reference(tmp_path, path), registration=stream['registration'], package=package['identity'],
                previous=None, arm='ordinary', target=0)
    first = train(tmp_path, data, device='cpu')
    (tmp_path / 'output').rename(tmp_path / 'zero')
    shutil.copytree(tmp_path / 'zero/checkpoint', tmp_path / 'inputs/previous', copy_function=os.link)
    second = train(tmp_path, dict(data, previous=first['checkpoint'], target=12), device='cpu')
    out = compare_next_update(tmp_path / 'output/checkpoint', package, second, device='cpu')
    assert out['parent_updates'] == 1 and out['reference'] == out['candidate']
    assert out['candidate']['updates'] == 2
    assert out['charged_tokens'] == 2 * (out['candidate']['tokens'] - second['milestone']['charged']) > 0
    assert out['candidate']['next_rows'] == digest(training_rows.matched_batch(stream, 64, 32))
    with pytest.raises(ValueError):
        compare_next_update(tmp_path / 'output/checkpoint', package, dict(second, milestone=first['milestone']), device='cpu')


def test_supervisor_loss_after_output_keeps_full_debit_and_reopens_without_redispatch(tmp_path):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import run_operation

    class SupervisorLoss(BaseException):
        pass

    grant = dict(identity=digest("operator"), holdback_us=5,
                 resources=dict(dict.fromkeys(KEYS, 0), attempts=1, elapsed_us=20_000_005))
    journal = BudgetJournal(tmp_path / "accounting", grant["identity"], grant["resources"])
    coordinates = dict(study=grant["identity"], stage="qualification", method="00", seed=42, look=0,
                       panel="training", profile="ordinary", input_mode="true", role="train", ordinal=0)
    produced = seal(dict(request=digest("frozen"), result=digest("producer bytes")))
    def interrupted(out, budget):
        assert budget == 20_000_000
        atomic_write(out / "producer.json", produced)
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        raise SupervisorLoss()

    args = dict(area=tmp_path, journal=journal, grant=grant, coordinates=coordinates, request=seal(dict(action="train")),
                cost={"attempts":1}, execute=interrupted, verify=lambda out: load(out / "producer.json"))
    with pytest.raises(SupervisorLoss):
        run_operation(**args)
    assert len(journal.snapshot()["pending"]) == 1
    first = run_operation(**args)
    assert first["evidence"] == produced and first["attempt"] == 0
    assert run_operation(**args) == first
    assert journal.snapshot()["remaining"]["elapsed_us"] == 5 and not journal.snapshot()["pending"]
