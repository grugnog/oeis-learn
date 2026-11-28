"""A registered synthetic donor exercises the normal run, journal and copied-report paths."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.velocity_metrics import case_outcome_receipt
from oeis_learn.pipeline_validation import progressive_round029 as command
from oeis_learn.pipeline_validation import progressive_velocity_run as runner
from oeis_learn.pipeline_validation import progressive_velocity_conditioning as conditioning
from oeis_learn.pipeline_validation import progressive_velocity_wave as wave
from oeis_learn.pipeline_validation import progressive_velocity_wave_report as report
from oeis_learn.pipeline_validation.progressive_evaluation import (
    model_receipt,
    probe_inference,
    velocity_probe,
)
from tests.integration.test_round029_comparative_wave import accepted_donor


@pytest.mark.parametrize("scenario", ["negative", "structural_only", "measured_zero"])
def test_installed_action_reopens_failure_and_copies_decision(tmp_path, monkeypatch, scenario):
    from oeis_learn.pipeline_validation import progressive_round028 as historical
    from oeis_learn.pipeline_validation import progressive_velocity_wave_host as host

    source = Path(__file__).resolve().parents[2]
    revision = "b" * 40
    from oeis_learn.generalization.progressive.cohorts import STRATA
    methods = [f"0{i}" for i in range(7)] if scenario == "negative" else ["00", "04", "05", "06"]
    strata = tuple(s for s in STRATA if scenario != "structural_only" or s != "oeis")
    folder, accepted, reg, packages = accepted_donor(tmp_path, source, methods, monkeypatch, strata=strata)
    seed = reg["initialization"]

    def prompt(row):
        offset = -3 if row["task"].endswith("0") else 7
        return dict(id=row["task"], indices=list(map(str, range(offset, offset + 20))),
                    values=[str(offset)] * 20)
    prompts = {r["task"]: prompt(r) for r in reg["development"]}
    price = dict(paired=accepted["sources"]["paired"], bank=accepted["sources"]["bank"], methods=methods,
                 expected_us=250, tail_us=310, search_us=120, selected_us=50, bank_us=60, overhead_us=20,
                 split_overhead_us={"ordinary-common": 1, "ordinary-full": 1, "D-full": 2})
    estimate = dict(sources=accepted["sources"], methods=methods, followup_reserve_us=accepted["followup_us"],
                    expected_us=accepted["expected_us"] + 17, tail_us=accepted["tail_us"] + 17,
                    items=dict(paired_search_selected_and_banks={k: price[k] for k in ("expected_us", "tail_us")}),
                    conditioning_components={k: price[k] for k in ("search_us", "selected_us", "bank_us", "overhead_us")},
                    conditioning_extra_role_us=price["split_overhead_us"])
    peak = {"preparation_us": runner.preparation.published(tmp_path, folder)["elapsed_us"],
            "checkpoint_read_us": 1000, "publication_us": 1_000_000, "prefixes": {}}
    estimate["provisional_publication"] = dict(checkpoint_read_us=peak["checkpoint_read_us"])
    planning = dict(verification_and_publication=dict(tail_us=peak["publication_us"]),
                    wave_space=dict(dispatch_prefixes=peak["prefixes"]))
    monkeypatch.setattr(command, "preflight", lambda *_: dict(inventory=dict(prices=dict(
        launch_estimate=estimate, launch_preparation=planning))))
    monkeypatch.setattr(host, "panel", lambda *_: prompts)
    sample = conditioning.freeze(accepted, reg, price, prompts, 1680 * 60_000_000, host_plan=peak)
    design = wave.plan(accepted, reg, packages, sample)
    from oeis_learn.pipeline_validation import progressive_velocity_wave_qualification as rehearsal
    qualified = digest("fixture installed worker qualification")
    with pytest.raises(ValueError, match="qualif"):
        rehearsal.required(tmp_path, folder, source, accepted)
    monkeypatch.setattr(rehearsal, "required", lambda *args: qualified)
    grant, approval, marker = wave.authority(
        tmp_path,
        revision,
        1680,
        accepted,
        design[-1],
        wave.driver_identity(source),
        wave.family_key(folder, create=True)["identity"], qualification=qualified,
    )
    monkeypatch.setattr(command, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(historical, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(
        historical,
        "git",
        lambda *a: velocity_policy()["donor_revision"] if a[1] == "rev-parse" else None,
    )
    with pytest.raises(ValueError, match="tail"):
        wave.authorize(tmp_path, source, 1, accepted["identity"])
    assert not (folder / conditioning.PATH).exists()
    path = wave.authorize(tmp_path, source, 1680, accepted["identity"])
    assert load(path) == grant
    qualification = runner.retained_qualification(tmp_path, grant)
    with pytest.raises(ValueError, match="original"):
        wave.authorize(tmp_path, source, 1681, accepted["identity"])
    monkeypatch.setattr(host, "next_files", lambda *a: [])
    calls = []
    failed = []
    failure_role = "score" if len(methods) == 7 else "train"

    class Roles:
        startup_us = 20_000_000
        training_tail_us = dict.fromkeys(methods, 600_000_000)
        incumbent = {
            "recipe": {},
            "origin": digest("D"),
            "weight": {"path": "inc.pt", "sha256": digest("weights"), "bytes": 1},
        }
        incumbent_path = tmp_path / "inc.pt"

        def __init__(self, *args, **kwargs):
            pass

        def credit_cut(self, method, out, cut):
            pass

        def channels(self, look):
            visible = [prompt(r) for r in look["members"]]
            return visible, [dict(p, values=p["values"] + ["123456789"] * 80) for p in visible]

        def supervise(self, request, data, extra, proof):
            if "barrier" in request:
                original = load(folder / "comparative" / grant["identity"][7:] / "sealed-conditioning-choices.json")
                assert request["barrier"] == original["identity"]
                assert len(original["choices"]) == 2 * (2 * len(methods) + 1)
            if request["role"] == "search":
                import json
                mounted = json.dumps([request, data])
                for member in [*reg["development"], *reg["retention"]]:
                    assert all(json.dumps(member[k]) not in mounted for k in ("task", "group", "stratum"))
                assert all(dst == "/work/inputs/model.pt" for _, dst in extra)
                assert "123456789" not in mounted

            def execute(out, allowance):
                role, method, target = (request[k] for k in ("role", "method", "target"))
                calls.append((role, method, target))
                atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
                if role == failure_role and method == "00" and target == 1_000_000 and not failed:
                    failed.append(True)
                    raise ValueError("fixture scorer interrupted after choices sealed")
                if role == "train":
                    key = "initial" if target == 0 else [method, target]
                    f = out / "learner/checkpoint/weight"
                    f.parent.mkdir(parents=True)
                    f.write_text(str(key))
                    weight = reference(f.parent, f)
                    model = model_receipt(
                        dict(registration=reg["identity"], method=method, seed=seed, look=target),
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
                        look, data["prompts"], mode=look["mode"], inference_seed=seed, model=model
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
                    import time
                    if method == "D" and scenario == "measured_zero":
                        time.sleep(0.06)
                    observations = {}
                    structural = {__import__("hmac").digest(bytes.fromhex(wave.family_key(folder)["key"]),
                                  b"case\0" + row["task"].encode(), "sha256").hex()
                                  for row in reg["development"] if row["stratum"] != "oeis"}
                    for k in probe["donors"]:
                        inference = probe_inference(probe)
                        success = (probe["mode"] == "true" if "barrier" in request else
                                   method == "D" if scenario == "negative" else
                                   method == "polynomial" or method == "04" and k in structural)
                        status = "solve" if success else "miss"
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
                if request["role"] == "train" and request["target"] == 1_000_000:
                    import time
                    time.sleep(.04)
                    assert reference(out / "learner/checkpoint", out / "learner/checkpoint/weight") == p["weight"]
                evidence = (
                    dict(cut=p["cut"], weight=p["weight"], files=[], allocations={})
                    if request["role"] == "train"
                    else proof(out, p)
                )
                return seal(dict(evidence, worker_cpu_us=1))

            return execute, verify, lambda: None

    monkeypatch.setattr(host, "Donor", Roles)
    monkeypatch.setattr(host, "capacity", lambda *args: None)
    monkeypatch.setattr(host, "custody", lambda *args: seal(dict(accepted=accepted["identity"])))
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
    from oeis_learn.generalization.progressive.trials import TrialJournal
    original_observe = TrialJournal.observe_velocity
    def lose_observation(self, registration, look, **kwargs):
        if look["method"] == "00" and look["look"] == 1_000_000:
            raise RuntimeError("fixture stopped before trial observation installation")
        return original_observe(self, registration, look, **kwargs)
    with monkeypatch.context() as fault:
        fault.setattr(TrialJournal, "observe_velocity", lose_observation)
        paused, code = command.execute(args, source)
        assert code == 1 and "before trial observation" in paused["reason"]
    if scenario == "measured_zero":
        original_supervise = Roles.supervise
        def stop_at_boundary(self, request, data, extra, proof):
            if request["target"] == 4_000_000:
                raise runner.InsufficientAuthority("next cut on mount 7: deficit 4096 bytes")
            return original_supervise(self, request, data, extra, proof)
        with monkeypatch.context() as fault:
            fault.setattr(Roles, "supervise", stop_at_boundary)
            checkpoint, code = command.execute(args, source)
            assert code == 2 and checkpoint["scientific_verdict"] == "inconclusive", checkpoint
            assert "deficit 4096 bytes" in checkpoint["reason"]
            verified, code = command.execute(SimpleNamespace(**dict(vars(args), action="verify", authority=None)), source)
            assert code == 0 and verified["report_ref"] == checkpoint["report_ref"], verified
            import shutil
            partial_copy, independent = tmp_path / "partial.zip", tmp_path / "partial-q.zip"
            shutil.copyfile(tmp_path / checkpoint["receipt_path"], partial_copy)
            shutil.copyfile(qualification, independent)
            hidden = folder.with_name("temporarily-unmounted")
            folder.rename(hidden)
            try:
                science, ledger = report.verify_release(partial_copy, independent)
            finally:
                hidden.rename(folder)
            assert science["decision"]["completed"] == 8
            outstanding = [r for r in science["decision"]["looks"] if r["reason"] == "registered_look_outstanding"]
            assert len(outstanding) == 12 and all(r["remaining_denominator"] == r["members"] > 0 for r in outstanding)
            assert ledger["remaining"]["elapsed_us"] == checkpoint["remaining_authority"]
            assert command.execute(args, source)[0]["report_ref"] == checkpoint["report_ref"]
    completed_searches = calls.count(("search", "00", 1_000_000))
    import time
    assemble = report.stage_archive
    def slow_cold_read(*args, **kwargs):
        if not kwargs["install"]:
            time.sleep(.04)
        return assemble(*args, **kwargs)
    monkeypatch.setattr(report, "stage_archive", slow_cold_read)
    publish = report.deliver

    def interrupted_publication(*args, **kwargs):
        publish(*args, **kwargs)
        raise RuntimeError("fixture interruption after ZIP installation")

    with monkeypatch.context() as fault:
        fault.setattr(report, "deliver", interrupted_publication)
        partial, code = command.execute(args, source)
        assert code == 1 and "after ZIP installation" in partial["reason"], partial
    settled_calls = len(calls)
    v, code = command.execute(args, source)
    assert code == 0 and v["scientific_verdict"] == (
        "retain_D" if len(methods) == 7 else "inconclusive"
    ), v
    assert (
        len(calls) == settled_calls and partial["remaining_authority"] == v["remaining_authority"]
    )
    assert calls[fixed] == (failure_role, "00", 1_000_000)
    assert calls.count(("search", "00", 1_000_000)) == completed_searches == 1
    assert len([r for r in calls if r[0] == "train"]) == 3 * len(methods) + (failure_role == "train")
    last = v["last_committed_ref"]
    count = len(calls)
    assert command.execute(args, source)[0]["last_committed_ref"] == last and len(calls) == count
    private = command.execute(
        SimpleNamespace(**dict(vars(args), action="verify", authority=None)), source
    )
    assert private[1] == 0 and private[0]["report_ref"] == last, private
    area = folder / ("comparative/" + grant["identity"][7:])
    assert load(area / "first-error.json")["stage"] == failure_role
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
    value, complete_account = report.verify_release(copy, qcopy)
    assert complete_account["remaining"]["elapsed_us"] == v["remaining_authority"]
    assert complete_account["consumed"]["elapsed_us"] == v["elapsed_us"]
    assert any(s["role"] == "train" and s["charged_us"] >= 40_000 for s in value["statistics"]["costs"].values())
    assert any(d["role"] == "publish" and d["work"]["elapsed_us"] >= 40_000 for d in complete_account["dispatches"])
    assert value["decision"]["excluded"] == accepted["omitted"]
    assert len(list((renamed / "registrations" / reg["identity"][7:]).glob("look-*.json"))) == 32
    for missing in value["decision"]["missing_looks"].values():
        assert len(missing) == 4 and all(r["remaining_denominator"] == r["denominator"] > 0 for r in missing)
    if failure_role == "train":
        point = value["curves"]["oeis"]["curves"]["00"]["looks"][1]
        assert point["issued_tokens"] == 2 * point["useful_tokens"]
        assert point["unknown_execution_tokens"] == point["useful_tokens"] and point["discarded_tokens"] == 0
    if scenario == "negative":
        assert value["direct"]["04"]["D"]["oeis"]["losses"] == 2
    else:
        candidate = next(r for r in value["decision"]["shortlist"] if r["method"] == "04")
        assert candidate["structural_pp"] == 100 and candidate["comparator"] == "00"
        assert candidate["proposed_branch"] == "oeis_measurement"
        missing = scenario == "structural_only"
        assert candidate["direct_oeis_pp"] == (None if missing else 0)
        assert candidate["measurement_supply"] == ("no_registered_oeis_families" if missing else "registered")
        assert candidate["marginal_us"] == (None if missing else accepted["followup_us"])
        if not missing:
            assert value["direct"]["polynomial"]["D"]["oeis"]["observed_pp"] == 100
            assert any(r["method"] == "polynomial" for r in value["decision"]["competitive_controls"])
    diagnostic = value["conditioning"]
    assert diagnostic["selected"]["00-common"]["oeis"]["observed_pp"] == (None if scenario == "structural_only" else 100)
    assert diagnostic["choices_barrier"] == load(renamed / "comparative" / grant["identity"][7:] /
                                                "sealed-conditioning-choices.json")["identity"]
    assert diagnostic["bank_dispatches"] == 0 and diagnostic["quote"] == price
    assert diagnostic["funding"] == sample["funding"]
    assert diagnostic["funding"]["extra_supervision_us"] == 2 * len(methods) + 2
    assert diagnostic["charged_search_and_selected_us"] > 0
    assert value["curves"]["retention"]["endpoints"]["D"]["yield"]["denominator"] == 1
    assert (
        value["curves"]["oeis"]["effects"]["4000000"]["batch_decay"]["interaction"]["denominator"]
        == (0 if scenario == "structural_only" else 2)
    )
    assert b'"task"' not in copy.read_bytes() and b'"truth"' not in copy.read_bytes()
    import json
    from copy import deepcopy

    records = report.read_canonical(copy, (report.MEMBER, report.ACCOUNT))
    account = json.loads(records[report.ACCOUNT])
    assert (
        account["consumed"]["elapsed_us"] + account["remaining"]["elapsed_us"] == grant["total_us"]
    )
    assert account["remaining"]["elapsed_us"] >= grant["holdback_us"]
    assert {d["role"] for d in account["dispatches"]} == {
        "prepare",
        "train",
        "search",
        "score",
        "publish",
    }
    if len(methods) == 7:
        scenario = deepcopy(value["statistics"])
        method_by_ref = {r["roster"]: r["method"] for r in accepted["schedule"]}
        panel = scenario["panels"]["full"]
        for family in panel["families"]:
            for pattern in family["patterns"]:
                for o, ref in zip(pattern["outcomes"], panel["rosters"], strict=True):
                    o["status"] = "solve" if method_by_ref[ref] == "polynomial" else "miss"
        for span in scenario["costs"].values():
            if span["role"] == "search" and span["beneficiaries"] in (["polynomial"], ["D"]):
                span["charged_us"] = 10 if span["beneficiaries"] == ["polynomial"] else 20
                span["end_us"] = span["start_us"] + span["charged_us"]
        alternative = report.reduce(accepted, scenario)["decision"]
        assert alternative["spending_action"] == "price_control_repeat"
        assert alternative["competitive_controls"][0]["method"] == "polynomial"
        assert alternative["incumbent"] == "D" and alternative["next_dispatch"] is None
        variant = seal(dict(value, statistics=scenario, **report.reduce(accepted, scenario)))
        billed = deepcopy(account)
        for dispatch in billed["dispatches"]:
            span = scenario["costs"].get(dispatch["receipt"])
            if span:
                diff = span["charged_us"] - dispatch["work"]["elapsed_us"]
                dispatch["work"]["elapsed_us"] += diff
                billed["consumed"]["elapsed_us"] += diff
                billed["remaining"]["elapsed_us"] -= diff
        alt = tmp_path / "fixed-control.zip"
        alt.write_bytes(report.archive_bytes({report.MEMBER: report.encode(variant),
                        report.ACCOUNT: report.encode(seal(dict(billed, report=variant["identity"])))}))
        with pytest.raises(ValueError, match="publication"):
            report.verify(alt, qcopy)
        # The underlying ledger check must catch a look assignment even without the signature gate.
        look = next(r for r in accepted["schedule"] if (r["method"],r["look"]) == ("00",1_000_000))
        bad_look = deepcopy(value)
        refs = bad_look["statistics"]["points"][look["roster"]]["work"]
        refs.remove(next(k for k in refs if value["statistics"]["costs"][k]["role"] == "search"))
        from oeis_learn.pipeline_validation import progressive_velocity_wave_commitment as commitment
        with pytest.raises(ValueError, match="primary look"):
            report.check_account(account, bad_look, commitment.read(qcopy))
        look = next(r for r in accepted["schedule"] if (r["method"], r["profile"]) == ("04", "full"))
        common = next(r for r in accepted["schedule"] if (r["method"], r["look"], r["profile"]) == ("04", 4_000_000, "common"))
        bad_look = deepcopy(value)
        bad_look["statistics"]["points"][look["roster"]]["work"] = value["statistics"]["points"][common["roster"]]["work"]
        with pytest.raises(ValueError, match="primary look"):
            report.check_account(account, bad_look, commitment.read(qcopy))
        for span in scenario["costs"].values():
            if span["beneficiaries"] in (["polynomial"], ["D"]) and span["role"] in {"search","score"}:
                charge = {"polynomial":{"search":20,"score":100}, "D":{"search":10,"score":10000}}
                span["charged_us"] = charge[span["beneficiaries"][0]][span["role"]]
                span["end_us"] = span["start_us"] + span["charged_us"]
        competitive = report.reduce(accepted, scenario)["decision"]["competitive_controls"][0]
        assert competitive["method"] == "polynomial" and competitive["competitive_axes"] == ["charged_us"]

    for i, tamper in enumerate(("decision", "count", "reserve", "parent", "family", "cut",
                                "larger_grant", "below_wave", "dispatch", "conditioning_scope")):
        changed, ledger = deepcopy(value), deepcopy(account)
        if tamper == "decision":
            changed["decision"]["verdict"] = "gain"
        elif tamper == "count":
            changed["statistics"]["panels"]["full"]["families"][0]["patterns"][0]["count"] += 1
        elif tamper == "reserve":
            ledger["remaining"]["elapsed_us"] += 1
        elif tamper == "parent":
            changed["accepted"] = digest("unrelated prescore")
        elif tamper == "family":
            cells = changed["statistics"]["panels"]["full"]["families"]
            a, b = (next(f for f in cells if f["stratum"] == s) for s in (strata[0], "retention"))
            a["family"], b["family"] = b["family"], a["family"]
        elif tamper == "cut":
            r = next(r for r in accepted["schedule"] if r["method"] == "00" and r["profile"] == "full")
            cut = changed["statistics"]["points"][r["roster"]]["cut"]
            for k in ("useful_tokens", "charged_tokens", "overshoot", "last_update_tokens"):
                cut[k] += 123456
        elif tamper == "dispatch":
            ledger["dispatches"][0]["attempt"] += 3
        elif tamper == "conditioning_scope":
            changed["statistics"]["conditioning"]["scope"]["donor_map"] = digest("different map")
            changed.update(report.reduce(accepted, changed["statistics"]))
        else:
            from oeis_learn.generalization.progressive.velocity_records import grant_terms
            total = 60_000_000 if tamper == "below_wave" else grant["total_us"] + 60_000_000
            terms = dict(grant_terms(grant), total_us=total,
                         resources=dict(grant["resources"], elapsed_us=total))
            approval = seal(dict(version="round029-authorization/v1", terms=digest(terms)))
            replacement = seal(dict(terms, authorizing_record_ref=approval["identity"]))
            ledger.update(grant=replacement, remaining=dict(ledger["remaining"],
                          elapsed_us=total-ledger["consumed"]["elapsed_us"]))
            changed.update(grant=replacement["identity"], controller=seal(dict(changed["controller"],
                           grant=replacement["identity"])))
        if tamper == "family":
            changed.update(report.reduce(accepted, changed["statistics"]))
        ledger["report"] = seal(changed)["identity"]
        bad = tmp_path / f"tampered-{i}.zip"
        bad.write_bytes(
            report.archive_bytes(
                {
                    report.MEMBER: report.encode(seal(changed)),
                    report.ACCOUNT: report.encode(seal(ledger)),
                }
            )
        )
        with pytest.raises(ValueError):
            report.verify(bad, qcopy)
