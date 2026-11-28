"""A registered synthetic donor exercises the normal run, journal and copied-report paths."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive import velocity_donation_report as public
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.velocity_metrics import case_outcome_receipt
from oeis_learn.generalization.progressive.velocity_records import SESSION
from oeis_learn.pipeline_validation import progressive_round029 as command
from oeis_learn.pipeline_validation import progressive_velocity_run as runner
from oeis_learn.pipeline_validation import progressive_velocity_wave as wave
from oeis_learn.pipeline_validation import progressive_velocity_wave_report as report
from oeis_learn.pipeline_validation.progressive_evaluation import (
    model_receipt,
    probe_inference,
    velocity_probe,
)
from tests.integration.test_round029_comparative_wave import fixture_wave


def test_installed_action_reopens_failure_and_copies_decision(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round028 as historical
    from oeis_learn.pipeline_validation import progressive_velocity_trial_preparation as preparation
    from oeis_learn.pipeline_validation import progressive_velocity_wave_host as host
    from oeis_learn.pipeline_validation.identity import file_hash

    source = Path(__file__).resolve().parents[2]
    revision = "b" * 40
    methods = ["00", "04", "05", "06"]
    accepted, reg, packages = fixture_wave(methods)
    # Role calls, outcomes and hardware identities are synthetic in this fixture.
    for m in methods:
        packages[m] = seal(dict(packages[m], profile={"model": {"seed": 9}}))
    training = [dict(r, learner=packages[r["method"]]["identity"]) for r in accepted["training"]]
    accepted = seal(
        dict(
            accepted,
            training=training,
            version=preparation.VERSION,
            seed=9,
            source=digest("source"),
            wheel=digest("wheel"),
            image=digest("image"),
            assessment="sealed",
            tail_us=90_000_000_000,
            preparation_grant=digest("original grant"),
            research_authority=None,
        )
    )
    folder = tmp_path / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    atomic_write(folder / "accepted.json", accepted)
    qualification = tmp_path / "external/qualification.zip"
    public.install(qualification, preparation.MEMBER, accepted)
    atomic_write(
        folder / ("qualification/" + accepted["preparation_grant"][7:] + "/grant.json"),
        {"original": 1},
    )
    (tmp_path / "run-state/discovery-authorizations").mkdir()
    design = wave.plan(accepted, reg, packages)
    grant, approval, marker = wave.authority(
        tmp_path, revision, 1680, accepted, design[-1], file_hash(source / wave.DRIVER)
    )
    monkeypatch.setattr(command, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(historical, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(
        historical,
        "git",
        lambda *a: velocity_policy()["donor_revision"] if a[1] == "rev-parse" else None,
    )
    monkeypatch.setattr(preparation, "published", lambda *_: {"accepted": accepted["identity"]})
    monkeypatch.setattr(wave, "inputs", lambda *_: (reg, packages))
    monkeypatch.setattr(runner, "retained_qualification", lambda *_: qualification)
    path = wave.authorize(tmp_path, source, 1680, accepted["identity"])
    assert load(path) == grant
    with pytest.raises(ValueError, match="original"):
        wave.authorize(tmp_path, source, 1681, accepted["identity"])
    monkeypatch.setattr(host, "next_files", lambda *a: [])
    calls = []
    failed = []

    class Roles:
        incumbent = {
            "recipe": {},
            "origin": digest("D"),
            "weight": {"path": "inc.pt", "sha256": digest("weights"), "bytes": 1},
        }
        incumbent_path = tmp_path / "inc.pt"

        def __init__(self, *args, **kwargs):
            pass

        def channels(self, look):
            prompts = [
                dict(id=r["task"], indices=list(map(str, range(20))), values=["0"] * 20)
                for r in look["members"]
            ]
            return prompts, []

        def supervise(self, request, data, extra, proof):
            def execute(out, allowance):
                role, method, target = (request[k] for k in ("role", "method", "target"))
                calls.append((role, method, target))
                atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
                if role == "score" and method == "00" and target == 1_000_000 and not failed:
                    failed.append(True)
                    raise ValueError("fixture scorer interrupted after choices sealed")
                if role == "train":
                    key = "initial" if target == 0 else [method, target]
                    f = out / "learner/checkpoint/weight"
                    f.parent.mkdir(parents=True)
                    f.write_text(str(key))
                    weight = reference(f.parent, f)
                    model = model_receipt(
                        dict(registration=reg["identity"], method=method, seed=9, look=target),
                        profile=digest(data["package"]["profile"]),
                        weights=digest(key),
                        checkpoint=weight["sha256"] if target else None,
                        origin=digest([method, target]),
                    )
                    p = dict(
                        cut=dict(
                            design[2][method][target],
                            charged_tokens=design[2][method][target]["useful_tokens"],
                            discarded_tokens=0,
                            checkpoint={},
                            target=target,
                            model=model,
                        ),
                        weight=weight,
                    )
                elif role == "search":
                    look = request["look"]
                    model = request["model"]
                    if method == "D":
                        model = model_receipt(
                            look,
                            profile=digest(data["recipe"]),
                            weights=digest("D"),
                            checkpoint=request["weight"]["sha256"],
                            origin=request["origin"],
                        )
                    probe = velocity_probe(
                        look, data["prompts"], mode="true", inference_seed=9, model=model
                    )
                    f = out / "generated/candidates.json"
                    atomic_write(
                        f,
                        seal(
                            dict(
                                version="discovery-candidates/v1",
                                clock=dict(velocity_probe=probe["identity"]),
                                records=dict.fromkeys(probe["donors"]),
                            )
                        ),
                    )
                    p = dict(probe=probe, artifact=reference(out, f))
                else:
                    probe = request["probe"]
                    observations = {}
                    for k in probe["donors"]:
                        inference = probe_inference(probe)
                        status = "solve" if method == "D" else "miss"
                        evidence = case_outcome_receipt(
                            k,
                            status=status,
                            outcome_ref=digest([inference, k]),
                            truth_ref=digest(k),
                            scorer_ref=digest("scorer"),
                            choice_ref=digest("choice"),
                            inference_ref=inference,
                        )
                        observations[k] = dict(
                            equal_budget=dict(
                                status=status,
                                closed=True,
                                complete=True,
                                probe_ref=probe["identity"],
                                receipt=evidence,
                            )
                        )
                    f = out / "scored/scores.json"
                    atomic_write(
                        f,
                        seal(
                            dict(
                                version="discovery-scoring/v1",
                                candidates=request["candidates"]["sha256"],
                                denominator=len(observations),
                                outcomes=observations,
                            )
                        ),
                    )
                    p = dict(artifact=reference(out, f))
                atomic_write(out / "producer.json", seal(p))

            def verify(out):
                p = load(out / "producer.json")
                evidence = (
                    {k: p[k] for k in ("cut", "weight")}
                    if request["role"] == "train"
                    else proof(out, p)
                )
                return seal(dict(evidence, worker_cpu_us=1))

            return execute, verify, lambda: None

    monkeypatch.setattr(host, "Donor", Roles)
    args = SimpleNamespace(
        action="run",
        workspace=str(tmp_path),
        authority=str(path),
        archive=None,
        qualification_archive=None,
    )
    v, code = command.execute(args, source)
    assert code == 1 and "fixture scorer interrupted" in v["reason"], v
    fixed = len(calls)
    v, code = command.execute(args, source)
    assert code == 0 and v["scientific_verdict"] == "inconclusive", v
    assert calls[fixed] == ("score", "00", 1_000_000)
    assert calls.count(("search", "00", 1_000_000)) == 1
    assert len([r for r in calls if r[0] == "train"]) == 12
    last = v["last_committed_ref"]
    count = len(calls)
    assert command.execute(args, source)[0]["last_committed_ref"] == last and len(calls) == count
    private = command.execute(
        SimpleNamespace(**dict(vars(args), action="verify", authority=None)), source
    )
    assert private[1] == 0 and private[0]["report_ref"] == last, private
    area = folder / ("comparative/" + grant["identity"][7:])
    assert load(area / "first-error.json")["stage"] == "score"
    import shutil

    copy = tmp_path / "copied.zip"
    shutil.copyfile(area / "share" / report.ARCHIVE, copy)
    qcopy = tmp_path / "copied-q.zip"
    shutil.copyfile(qualification, qcopy)
    renamed = tmp_path / "unmounted"
    folder.rename(renamed)
    verify_args = SimpleNamespace(
        action="verify",
        workspace=None,
        authority=None,
        archive=str(copy),
        qualification_archive=str(qcopy),
    )
    result, code = command.execute(verify_args, source)
    assert code == 0 and result["report_ref"] == last, result
    value = public.read_member(copy, report.MEMBER)
    assert value["decision"]["excluded"] == accepted["omitted"]
    assert value["direct"]["04"]["D"]["oeis"]["losses"] == 1
    assert value["curves"]["retention"]["endpoints"]["D"]["yield"]["denominator"] == 1
    assert (
        value["curves"]["oeis"]["effects"]["4000000"]["batch_decay"]["interaction"]["denominator"]
        == 1
    )
    assert b'"task"' not in copy.read_bytes() and b'"truth"' not in copy.read_bytes()
    bad = tmp_path / "tampered.zip"
    value["decision"]["verdict"] = "gain"
    public.install(bad, report.MEMBER, seal(value))
    with pytest.raises(ValueError):
        report.verify(bad, qcopy)
