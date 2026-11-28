"""Exercise the effective installed comparative roles on disposable inputs before exploration."""

from pathlib import Path

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest, load, read_bound_record, reference, seal
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id, validate_grant
from . import progressive_velocity_wave as wave
from . import progressive_velocity_wave_host as physical
from .identity import confined, file_hash
from .preparation_recovery import retain
from .progressive_velocity_broad_host import attempt
from .progressive_velocity_operations import run_operation
from .progressive_velocity_qualification import authorize_qualification
from .progressive_velocity_trial_preparation import accepted_preparation


def binding(source, accepted):
    return seal(dict(version="round029-worker-qualification/v1", accepted=accepted["identity"],
                     driver=wave.driver_identity(source), wheel=accepted["wheel"], source=accepted["source"],
                     image=accepted["image"], physical=accepted["physical"],
                     supervisor=file_hash(Path(physical.__file__)),
                     qualification=file_hash(Path(__file__))))


def directory(folder, bound):
    return confined(folder, "comparative-worker/" + bound["identity"][7:], exists=False)


def required(root, folder, source, accepted):
    bound = binding(source, accepted)
    location = directory(folder, bound)
    if not (location / "qualified.json").is_file():
        raise ValueError("effective comparative worker requires a funded installed qualification")
    record = load(confined(location, "qualified.json"))
    grant = load(confined(location, "grant.json"))
    approval = load(confined(root, "run-state/discovery-authorizations/" +
                             grant["authorizing_record_ref"][7:] + ".json"))
    validate_grant(grant, approval, root=physical_root_id(root), parent=bound["identity"], action="qualify")
    paid = BudgetJournal.inspect_existing(location / "accounting", grant["identity"], grant["resources"])
    if (record != seal(record) or record["binding"] != bound or record["grant"] != grant["identity"]
            or paid["pending"] or record["consumed"] != paid["consumed"]
            or set(record["operations"]) != {"train", "search", "score"}):
        raise ValueError("original comparative worker qualification or paid balance differs")
    for role, key in record["operations"].items():
        path = confined(location, "operations/" + key[7:])
        op = load(confined(path, "receipt.json"))
        raw = load(confined(path, "input.json"))
        if (raw["coordinates"]["role"] != role or op != seal(op)
                or op["identity"] != record["receipts"][role]
                or key + "/dispatch/" + str(op["attempt"]) not in paid["committed"]):
            raise ValueError("original paid worker proof differs")
        producer = load(confined(path, f"attempt-{op['attempt']:04d}/producer.json"))
        if producer != seal(producer) or op["evidence"]["producer"] != producer["identity"]:
            raise ValueError("qualified producer differs from its retained proof")
    return record["identity"]


def terms(root, folder, accepted):
    reg, packages = wave.inputs(root, folder, accepted)
    learner = packages["00"]
    first = wave.targets(learner, [1])[1]
    member = dict(task="disposable-index-plus-two", group="exposed-affine", stratum="parameter")
    look = velocity_look(registration=reg["identity"], method="polynomial", seed=reg["initialization"],
                         look=4_000_000, profile="full", development=[member], retention=[])
    costs = dict(train=dict(tokens=first["useful_tokens"], attempts=1),
                 search=wave.probe_work(look, "search"), score=wave.probe_work(look, "score"))
    return learner, first, look, costs


def authorize(root, source, minutes, expected):
    folder = confined(root, "run-state/discovery/" + SESSION)
    with RunStore.owner(folder):
        accepted = accepted_preparation(folder)
        if not accepted or accepted["identity"] != expected:
            raise ValueError("operator's accepted wave differs")
        bound = binding(source, accepted)
        _, _, _, costs = terms(root, folder, accepted)
        work = {k: sum(c.get(k, 0) for c in costs.values()) + max(c.get(k, 0) for c in costs.values())
                for k in set().union(*costs.values())}
        path = authorize_qualification(root, minutes, bound["identity"], work)
        retain(directory(folder, bound) / "grant.json", load(path))
        return path


def qualify(args, root, folder, source, revision, grant, host_factory):
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from .progressive_round029 import preflight, result
    accepted = accepted_preparation(folder)
    bound = binding(source, accepted)
    area = directory(folder, bound)
    if load(confined(area, "grant.json")) != grant or grant["registration_or_parent_ref"] != bound["identity"]:
        raise ValueError("original worker qualification grant required")
    path = area / "qualified.json"
    if not path.is_file():
        learner, first, look, costs = terms(root, folder, accepted)
        wheel = reference(root, physical.runtime(root, folder, accepted))
        basis = seal(dict(accepted=accepted["identity"], wheel=wheel, capacity_owners={},
                          startup_us=0, training_tail_us={}, witnesses={}, incumbent=None))
        donor = physical.Donor(root, source, area, accepted, basis, revision, args, host_factory)
        prices = preflight(root, source)["inventory"]["prices"]["launch_preparation"]
        donor.prefixes = prices["wave_space"]["dispatch_prefixes"]
        journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
        indices = list(range(-3, 97))
        truth = dict(id=look["members"][0]["task"], indices=list(map(str, indices)),
                     values=[str(i + 2) for i in indices])
        prompt = dict(id=truth["id"], indices=truth["indices"][:20], values=truth["values"][:20])
        view, visible, handles = physical.visible_inputs(look, [prompt], bytes.fromhex(bound["identity"][7:]), "true")
        base = dict(version=wave.VERSION, seed=look["seed"], registration=look["registration"],
                    source=accepted["source"], wheel=accepted["wheel"], image=accepted["image"],
                    driver=bound["driver"], accepted=accepted["identity"])
        records, keys = {}, {}
        for ordinal, role in enumerate(("train", "search", "score")):
            method = "00" if role == "train" else "polynomial"
            coords = dict(study=grant["identity"], stage="worker_qualification", method=method,
                          seed=look["seed"], look=1 if role=="train" else 4_000_000,
                          panel="disposable", profile="full", input_mode="true", role=role, ordinal=ordinal)
            kw = dict(base, role=role, method=method, target=coords["look"], gpu=role == "train",
                      coordinate=digest(coords))
            extra = []
            if role == "train":
                data = dict(package=learner)
                kw.update(learner=learner["identity"], cap=first["useful_tokens"], parent_files=[],
                          qualify_restore=True)
                def proof(out, producer):
                    cut = physical.check_cut(out, request, learner, producer, first)["cut"]
                    restored = producer["restore"]
                    if (producer["device"] != "hip" or cut["updates"] != 1 or cut["weights"] == learner["initial_weights"]
                            or restored["written"] != restored["reopened"]):
                        raise ValueError("the installed HIP update and restore did not qualify")
                    return dict(restore=restored, weights=cut["weights"])
            elif role == "search":
                data = dict(recipe=learner["profile"], prompts=visible)
                kw.update(look=view, model=None, weight=None, seconds=grant["total_us"] // 1_000_000)
                def proof(out, producer):
                    ref = producer["artifact"]
                    generated = read_bound_record(out, ref, version="discovery-candidates/v1")
                    if generated["clock"]["velocity_probe"] != producer["probe"]["identity"]:
                        raise ValueError("qualification choice differs from its registered probe")
                    return dict(probe=producer["probe"], artifact=ref)
            else:
                ref = records["search"]["evidence"]["artifact"]
                extra = [(physical.checked_file(search_out, ref), "/work/inputs/candidates.json")]
                data = dict(truth=[dict(truth, id=handles[truth["id"]])])
                kw.update(probe=records["search"]["evidence"]["probe"], candidates=dict(ref, path="inputs/candidates.json"),
                          seconds=grant["total_us"] // 1_000_000)
                def proof(out, producer):
                    scores = read_bound_record(out, producer["artifact"], version="discovery-scoring/v1")
                    if (scores["candidates"] != ref["sha256"] or scores["denominator"] != 1
                            or scores["outcomes"][handles[truth["id"]]]["equal_budget"]["status"] != "solve"):
                        raise ValueError("installed native scorer failed the exposed affine witness")
                    return dict(score=scores["identity"])
            request = seal(dict(kw, data=digest(data)))
            execute, verify, guard = donor.supervise(request, data, extra, proof)
            records[role] = run_operation(area=area, journal=journal, grant=grant, coordinates=coords,
                request=request, cost=costs[role], execute=execute, verify=verify, before_dispatch=guard,
                minimum_us=121_000_000, maximum_us=grant["total_us"] // 2, max_attempts=2)
            keys[role] = operation_key(coords, 0)["logical"]
            if role == "search":
                search_out = attempt(area, coords, grant)
        retain(path, seal(dict(binding=bound, grant=grant["identity"], operations=keys,
                               receipts={k:v["identity"] for k,v in records.items()}, consumed=journal.snapshot()["consumed"])))
    ref = required(root, folder, source, accepted)
    record = load(path)
    return result("qualify", "COMPLETE", "Installed worker updated and restored on HIP and scored its exposed witness",
                  "run", last=ref, remaining=grant["total_us"]-record["consumed"]["elapsed_us"],
                  receipt=record, receipt_path=str(path.relative_to(root))), 0
