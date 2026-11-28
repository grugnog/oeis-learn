"""Run or reopen the earliest eligible comparative operation under one session owner."""

from copy import deepcopy
import sys
import time

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load, read_bound_record, seal
from oeis_learn.generalization.progressive.velocity_metrics import measured_look
from oeis_learn.generalization.progressive.velocity_records import operation_key
from . import progressive_velocity_trial_preparation as preparation
from . import progressive_velocity_wave as wave
from . import progressive_velocity_wave_host as physical
from .identity import confined, file_hash
from .preparation_recovery import retain, revise
from .progressive_evaluation import probe_inference, velocity_probe
from .progressive_research_inputs import checked_file
from .progressive_velocity_broad_host import attempt
from .progressive_velocity_donation import retained_qualification
from .progressive_velocity_operations import (
    InsufficientAuthority,
    funded_verification,
    replay_operation,
    run_operation,
)


def original_balance(area, grant):
    return BudgetJournal.inspect_existing(
        area / "accounting",
        grant["identity"],
        grant["resources"],
        include_work=True,
        include_windows=True,
    )


def window(grant, accepted, coords, request, donor, cost):
    if coords["stage"] != "exploration":
        return grant["total_us"] - grant["holdback_us"]
    method = coords["method"]
    if coords["role"] == "train":
        tail = (donor.training_tail_us[method] * cost["tokens"] + 3_999_999) // 4_000_000
    else:
        look = request["look"] if coords["role"] == "search" else request["probe"]["look"]
        # The paid matched follow-up covers two complete OEIS searches and their scoring.
        # Also preserve each registered issuance and compact-fit drain opportunity.
        per_case = max(
            accepted["followup_us"],
            look["opportunity"]["issue_us"] + look["attempts"] * look["opportunity"]["fit_us"],
        )
        tail = len(look["members"]) * per_case
    return 120_000_000 + 2 * (donor.startup_us + tail)


def costs(snapshot, grant, coords, op, names):
    prefix = operation_key(coords, 0)["logical"] + "/"
    windows = snapshot["windows"]
    rows = []
    for key in sorted(k for k in snapshot["committed"] if k.startswith(prefix)):
        w = windows[key]
        certified = key == prefix + "dispatch/" + str(op["attempt"])
        proof = op["evidence"] if certified else {}
        device = proof.get("telemetry", {})
        rows.append(
            dict(
                origin="incremental",
                receipt=digest([grant["identity"], key]),
                role=coords["role"],
                beneficiaries=names,
                state="complete" if certified else "failed",
                host=grant["state_root_id"],
                boot=w["boot"],
                start_us=w["start_us"],
                end_us=w["end_us"],
                charged_us=w["charged_us"],
                issued_tokens=snapshot["reserved_work"][key]["tokens"],
                worker_cpu_us=proof.get("worker_cpu_us"),
                lease_us=proof.get("lease_us"),
                covered_us=device.get("covered_us"),
                busy_us=device.get("busy_us"),
            )
        )
    return rows


def alias(original, look, prompts, cut):
    probe = velocity_probe(
        look,
        prompts,
        mode="true",
        inference_seed=original["probe"]["inference_seed"],
        model=cut["model"],
    )
    if probe_inference(probe) != probe_inference(original["probe"]):
        raise ValueError("zero-look reuse requires exactly the same inference opportunity")
    scores = deepcopy(original["scores"])
    for obs in scores["outcomes"].values():
        obs["equal_budget"]["probe_ref"] = probe["identity"]
    return measured_look(probe, seal(scores), cut=cut, work=original["work"])


def screen(area, grant, marker, accepted, packages, design, donor, invoke, progress, observe):
    _, looks, crossings, ops = design
    cuts, generated, observed, train_work, common_work = {}, {}, {}, {}, {}
    shared = None
    for ordinal, op in enumerate(ops):
        look = looks[op["roster"]]
        method, target, profile, role = look["method"], look["look"], look["profile"], op["role"]
        progress(method, target, role)
        learner = packages.get(method)
        coords = dict(
            study=grant["identity"],
            stage="exploration",
            method=method,
            seed=look["seed"],
            look=target,
            panel="development",
            profile=profile,
            input_mode="true",
            role=role,
            ordinal=ordinal,
        )
        key = method, target
        cut = cuts.get(key)
        common = dict(
            version=wave.VERSION,
            role=role,
            method=method,
            target=target,
            seed=look["seed"],
            registration=accepted["registration"],
            source=accepted["source"],
            image=accepted["image"],
            wheel=accepted["wheel"],
            driver=marker["driver"],
            accepted=accepted["identity"],
            coordinate=digest(coords),
        )
        extra = []
        if role == "train":
            previous = [t for m, t in cuts if m == method and t < target]
            parent_files = []
            if previous:
                parent, out = cuts[method, max(previous)]
                parent_dir = confined(out, "learner")
                parent_files = physical.next_files(parent_dir, parent["cut"]["checkpoint"])
                extra = [
                    (checked_file(parent_dir, f), "/work/inputs/parent/" + f["path"])
                    for f in parent_files
                ]
            data = dict(package=learner)
            max_useful = crossings[method][max(crossings[method])]["useful_tokens"]
            request = seal(
                dict(
                    common,
                    data=digest(data),
                    gpu=True,
                    parent_files=parent_files,
                    learner=learner["identity"],
                    cap=max_useful
                    + max(
                        o["cost"]["tokens"]
                        for o in ops
                        if o["role"] == "train" and looks[o["roster"]]["method"] == method
                    ),
                )
            )

            def proof(out, producer):
                return physical.check_cut(
                    out, request, learner, producer, crossings[method][target]
                )

        else:
            prompts, truth = donor.channels(look)
            if target == 0 and shared is not None:
                if role == "score":
                    m = alias(shared, look, prompts, cut[0]["cut"])
                    # Each learner owns its initialization. All beneficiaries share the one probe.
                    shared_roles = [r for r in m["work"] if r["role"] != "train"]
                    observed[look["identity"]] = measured_look(
                        m["probe"],
                        m["scores"],
                        cut=cut[0]["cut"],
                        work=train_work[method] + shared_roles,
                    )
                    observe(look, observed[look["identity"]])
                    common_work.setdefault(method, []).extend(shared_roles)
                continue
            if role == "search":
                recipe = learner["profile"] if learner else donor.incumbent["recipe"]
                weight, model, origin = None, None, None
                if cut:
                    evidence, output = cut
                    model = evidence["cut"]["model"]
                    if target:
                        f = evidence["weight"]
                        weight = dict(f, path="inputs/model.pt")
                        extra = [
                            (
                                checked_file(confined(output, "learner/checkpoint"), f),
                                "/work/inputs/model.pt",
                            )
                        ]
                elif method == "D":
                    origin = donor.incumbent["origin"]
                    weight = dict(donor.incumbent["weight"], path="inputs/model.pt")
                    extra = [(donor.incumbent_path, "/work/inputs/model.pt")]
                data = dict(prompts=prompts, recipe=recipe)
                request = seal(
                    dict(
                        common,
                        data=digest(data),
                        gpu=method not in velocity_policy()["controls"],
                        model=model,
                        weight=weight,
                        origin=origin,
                        look=look,
                        seconds=grant["total_us"] // 1_000_000,
                    )
                )

                def proof(out, producer):
                    probe = producer["probe"]
                    actual_model = probe["model"] if method == "D" else model
                    if method == "D" and (
                        actual_model["origin"] != origin
                        or actual_model["profile"] != digest(recipe)
                        or actual_model["checkpoint"] != weight["sha256"]
                    ):
                        raise ValueError("incumbent differs from its frozen weight slice")
                    if probe != velocity_probe(
                        look,
                        prompts,
                        mode="true",
                        inference_seed=request["seed"],
                        model=actual_model,
                    ):
                        raise ValueError("visible inference differs from the frozen opportunity")
                    ref = producer["artifact"]
                    if ref["path"] != "generated/candidates.json":
                        raise ValueError("unexpected generated choice artifact")
                    choices = read_bound_record(out, ref, version="discovery-candidates/v1")
                    if choices["clock"]["velocity_probe"] != probe["identity"] or set(
                        choices["records"]
                    ) != set(probe["donors"]):
                        raise ValueError("sealed choice roster differs")
                    return dict(artifact=ref, probe=probe)

            else:
                search, search_out, _ = generated[look["identity"]]
                probe, f = search["probe"], search["artifact"]
                reference = dict(f, path="inputs/candidates.json")
                extra = [(checked_file(search_out, f), "/work/inputs/candidates.json")]
                data = dict(truth=truth)
                request = seal(
                    dict(
                        common,
                        data=digest(data),
                        gpu=False,
                        probe=probe,
                        candidates=reference,
                        seconds=grant["total_us"] // 1_000_000,
                    )
                )

                def proof(out, producer):
                    ref = producer["artifact"]
                    if ref["path"] != "scored/scores.json":
                        raise ValueError("unexpected selected-score artifact")
                    scores = read_bound_record(out, ref, version="discovery-scoring/v1")
                    if scores["candidates"] != reference["sha256"]:
                        raise ValueError("scorer used different sealed choices")
                    measured_look(probe, scores, cut=cut[0]["cut"] if cut else None)
                    return dict(artifact=ref)

        execute, verify, guard = donor.supervise(request, data, extra, proof)
        paid, out, snapshot = invoke(coords, request, op["cost"], execute, verify, guard)
        names = accepted["supported"] if target == 0 and role != "train" else [method]
        work = costs(snapshot, grant, coords, paid, names)
        evidence = paid["evidence"]
        if role == "train":
            cuts[key] = evidence, out
            donor.credit_cut(method, out, evidence["cut"])
            train_work.setdefault(method, []).extend(work)
        elif role == "search":
            generated[look["identity"]] = evidence, out, work
        else:
            scores = read_bound_record(out, evidence["artifact"], version="discovery-scoring/v1")
            probe_work = generated[look["identity"]][2] + work
            cumulative = train_work.get(method, []) + common_work.get(method, []) + probe_work
            m = measured_look(probe, scores, cut=cut[0]["cut"] if cut else None, work=cumulative)
            observed[look["identity"]] = m
            observe(look, m)
            if profile == "common":
                common_work.setdefault(method, []).extend(probe_work)
            if target == 0:
                shared = m
    return observed


def run(args, root, folder, source, revision, grant, host_factory, *, readonly=False):
    from .progressive_round029 import result
    from . import progressive_velocity_wave_report as report

    accepted = preparation.accepted_preparation(folder)
    if not accepted or grant["scope"] != "exploration":
        raise ValueError(
            "funded supported exploration requires its independent pre-score commitment"
        )
    preparation.published(root, folder)
    reg, packages = wave.inputs(root, folder, accepted)
    design = wave.plan(accepted, reg, packages)
    minutes, rest = divmod(grant["total_us"], 60_000_000)
    secret = wave.family_key(folder)
    expected, _, marker = wave.authority(
        root,
        revision,
        minutes,
        accepted,
        design[-1],
        file_hash(confined(source, wave.DRIVER)),
        secret["identity"],
    )
    if rest or grant != expected or load(confined(folder, wave.MARKER)) != marker:
        raise ValueError("controller, driver or cumulative research grant differs")
    area = confined(folder, "comparative/" + grant["identity"][7:], exists=readonly)
    if not readonly:
        area.mkdir(parents=True, exist_ok=True)
        retain(area / "grant.json", grant)
        journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    elif grant != load(confined(area, "grant.json")):
        raise ValueError("original comparative grant differs")
    donor = None
    status = dict(recipe=None, look=None, stage="preparation")

    def progress(method, target, phase):
        status.update(recipe=method, look=target, stage=phase)

    def invoke(coords, request, cost, execute, verify, guard):
        if readonly:
            snapshot = original_balance(area, grant)
            paid = replay_operation(area, coords, request, grant, cost, snapshot, verify)
        else:
            value = result(
                "run",
                "PARTIAL",
                "Comparative wave in progress",
                "run",
                **status,
                seed=accepted["seed"],
                remaining=journal.snapshot()["remaining"]["elapsed_us"],
            )
            revise(folder / "status.json", seal(value))
            print(
                f"round029 {coords['role']} {coords['method']} {coords['look']} {coords['profile']}",
                file=sys.stderr,
                flush=True,
            )
            paid = run_operation(
                area=area,
                journal=journal,
                grant=grant,
                coordinates=coords,
                request=request,
                cost=cost,
                execute=execute,
                verify=verify,
                before_dispatch=guard,
                max_attempts=2,
                minimum_us=121_000_000 if coords["stage"] == "exploration" else 1,
                maximum_us=window(grant, accepted, coords, request, donor, cost),
            )
            snapshot = original_balance(area, grant)
        return paid, attempt(area, coords, grant), snapshot

    try:
        from .progressive_velocity_image import preparation_boundary

        def host_space():
            physical.capacity(area, accepted, getattr(args, "free_disk_gib", 5) << 30,
                              getattr(donor, "capacity_owners", None), getattr(donor, "retained", None),
                              status["stage"])

        def custody_basis(out, *, install):
            basis = physical.custody(root, folder, accepted, reg, source)
            if install:
                retain(out / "custodian.json", basis)
            if load(confined(out, "custodian.json"), maximum=wave.MAX_INPUT) != basis:
                raise ValueError("registered custodian material differs")
            return seal(dict(inputs=basis["identity"]))

        def prepare(out, allowance):
            deadline = time.monotonic_ns() // 1000 + allowance
            with preparation_boundary(out, deadline * 1000):
                retain(
                    out / "producer.json",
                    funded_verification(
                        lambda dest: custody_basis(dest, install=True), out, deadline
                    ),
                )

        def verify_basis(out):
            basis = custody_basis(out, install=False)
            if load(confined(out, "producer.json")) != basis:
                raise ValueError("original custodian receipt differs")
            return basis

        coords = dict(
            study=grant["identity"],
            stage="preparation",
            method="wave",
            seed=accepted["seed"],
            look=0,
            panel="development",
            profile="full",
            input_mode="true",
            role="prepare",
            ordinal=0,
        )
        request = seal(
            dict(grant=grant["identity"], accepted=accepted["identity"], controller=revision)
        )
        _, private, _ = invoke(coords, request, {}, prepare, verify_basis, host_space)
        basis = load(confined(private, "custodian.json"), maximum=wave.MAX_INPUT)
        donor = physical.Donor(
            root, source, area, accepted, basis, revision, args, host_factory, readonly=readonly
        )
        from oeis_learn.generalization.progressive.trials import TrialJournal

        registry = confined(folder, "registrations/" + reg["identity"][7:])
        trials = TrialJournal(registry, study=reg["identity"], readonly=readonly)
        for method, expected_trial in accepted["trials"].items():
            if trials.velocity_trial(method)["identity"] != expected_trial:
                raise ValueError("research observation belongs to another frozen trial")
        for method, reason in accepted["omitted"].items():
            for series in design[0][method].values():
                for look in series:
                    trials.observe_velocity(reg, look, reason=reason)
        def observe(look, measurement):
            trials.observe_velocity(reg, look, evidence=measurement)
        observed = screen(area, grant, marker, accepted, packages, design, donor, invoke, progress, observe)
        status.update(stage="publication", recipe=None, look=None)
        qualification = retained_qualification(root, grant)
        request = seal(
            dict(
                accepted=accepted["identity"],
                grant=grant["identity"],
                evidence=digest({k: v["identity"] for k, v in observed.items()}),
            )
        )
        coords = dict(
            study=grant["identity"],
            stage="publication",
            method="wave",
            seed=accepted["seed"],
            look=4_000_000,
            panel="development",
            profile="full",
            input_mode="true",
            role="publish",
            ordinal=0,
        )

        def public_proof(out):
            record = report.publication(
                area, grant, accepted, observed, marker=marker, key=secret["key"], readonly=True
            )
            if record != report.checked_science(record, qualification):
                raise ValueError("copied comparative report differs from private observations")
            return seal(dict(report=record["identity"], verdict=record["decision"]["verdict"]))

        def publish(out, allowance):
            deadline = time.monotonic_ns() // 1000 + allowance
            with preparation_boundary(out, deadline * 1000):

                def install(_):
                    record = report.publication(
                        area,
                        grant,
                        accepted,
                        observed,
                        marker=marker,
                        key=secret["key"],
                        readonly=False,
                    )
                    return seal(
                        dict(report=record["identity"], verdict=record["decision"]["verdict"])
                    )

                retain(out / "producer.json", funded_verification(install, out, deadline))

        def verify_publication(out):
            proof = public_proof(out)
            if load(confined(out, "producer.json")) != proof:
                raise ValueError("published report differs from its original grant receipt")
            return proof

        paid, _, balance = invoke(coords, request, {}, publish, verify_publication, host_space)
        record = paid["evidence"]
        settled = report.deliver(area, grant, accepted, balance, qualification=qualification, readonly=readonly)
        if settled["report"] != record["report"]:
            raise ValueError("installed final accounting belongs to another publication")
        value = result(
            "verify" if readonly else "run",
            "VERIFIED" if readonly else "COMPLETE",
            "Comparative exploration and independent archive replay complete",
            "review_exploration",
            last=record["report"],
            report_ref=record["report"],
            remaining=balance["remaining"]["elapsed_us"],
            **(
                dict(verification_scope="private_comparative_wave")
                if readonly
                else dict(
                    elapsed_us=balance["consumed"]["elapsed_us"],
                    charged_tokens=balance["consumed"]["tokens"],
                    receipt_path=str((area / "share" / report.ARCHIVE).relative_to(root)),
                    qualification_path=str(qualification.relative_to(root)),
                )
            ),
        )
        value["scientific_verdict"] = record["verdict"]
        code = 0
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        if readonly:
            raise
        reason = str(exc) or type(exc).__name__
        error = dict(stage=status["stage"], error=type(exc).__name__, reason=reason)
        first = area / "first-error.json"
        if not first.exists():
            retain(first, error)
        code = 2 if isinstance(exc, InsufficientAuthority) else 1
        balance = journal.snapshot()
        value = result(
            "run",
            "BLOCKED" if code == 2 else "PARTIAL",
            reason,
            "resume_original_grant",
            remaining=balance["remaining"]["elapsed_us"],
            elapsed_us=balance["consumed"]["elapsed_us"],
            charged_tokens=balance["consumed"]["tokens"],
            first_error=load(first),
            **status,
        )
    if not readonly:
        revise(folder / "status.json", seal(value))
    return value, code
