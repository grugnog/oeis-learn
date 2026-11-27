"""Installed CPU comparison of an immutable bank, followed by independent live searches."""

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import time

from oeis_learn.generalization.progressive.contracts import (
    digest,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_selector import (
    choose,
    fit_ranker,
    shuffled_values,
    SELECTOR_ARMS, COMPONENTS, selector_spending, selector_counts, verify_selector,
)
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require

ARM_NAMES = SELECTOR_ARMS[:3]
LIVE_NAMES = SELECTOR_ARMS[3:]
COMPONENT_NAMES = COMPONENTS
CONVENTIONAL = {
    "template": "templates",
    "polynomial": "polynomial",
    "linear_recurrence": "linear_recurrence",
}


def historical_reference(cut, *, trial_eligible):
    """Resolve the registered tie rule under every completion of missing primary verdicts."""
    names = ["retained_D", *CONVENTIONAL]
    if trial_eligible:
        names.append("trial05")
    bounds = {}
    for name in names:
        label = (
            "retained_D/" + name
            if name in CONVENTIONAL
            else "trial-05-1M"
            if name == "trial05"
            else name
        )
        families = defaultdict(Counter)
        for case in cut["cases"]:
            if case["stratum"] in {"composition", "unseen_structure"}:
                observation = case["observations"].get(label, {})
                verdict = observation.get("frozen")
                families[(case["stratum"], case["group"])][verdict] += 1
        require(families, "historical primary families required")
        totals = defaultdict(lambda: [Fraction(), Fraction(), 0])
        unknown = 0
        for (stratum, _), counts in families.items():
            total = sum(counts.values())
            missing = total - counts["success"] - counts["failure"]
            unknown += missing
            totals[stratum][0] += Fraction(counts["success"], total)
            totals[stratum][1] += Fraction(counts["success"] + missing, total)
            totals[stratum][2] += 1
        for stratum in {"composition", "unseen_structure"} - totals.keys():
            # No observed family constrains this stratum of the registered 32-case roster.
            totals[stratum] = [Fraction(0), Fraction(1), 1]
            unknown += 32
        lower = sum(low / n for low, _, n in totals.values()) / 2
        upper = sum(high / n for _, high, n in totals.values()) / 2
        bounds[name] = (lower, upper, unknown)
    best = min(names, key=lambda name: (-bounds[name][0], bounds[name][2], name))
    ambiguous = any(
        (-bounds[other][1], bounds[other][2], other) < (-bounds[best][0], bounds[best][2], best)
        for other in names
        if other != best
    )
    return dict(
        method="retained_D" if ambiguous else best,
        ambiguous=ambiguous,
        bounds={
            key: [str(low), str(high), unknown] for key, (low, high, unknown) in bounds.items()
        },
    )


def eligible_sources(bank, prompt):
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
    from oeis_learn.synthesis.admission import source_identity, VisiblePrompt

    expected = VisiblePrompt(tuple(prompt["indices"][:20]), tuple(prompt["values"][:20])).sha256
    sources = {}
    for row in bank:
        selected = validate_selection_envelope(row)
        key = selected["source_sha256"]
        require(
            key == source_identity(selected["source"])
            and selected["verified"] is True
            and selected["visible_count"] == 20
            and row["prompt_sha256"] == expected
            and key not in sources,
            "eligible selector source or true prefix differs",
        )
        sources[key] = row
    return sources


def selector_choice(data, client):
    from oeis_learn.fresh.bigint_diagnostics import portfolio, model_snapshot
    from oeis_learn.generalization.progressive.baselines import load_baseline
    from oeis_learn.generalization.progressive.cohorts import validate_prompt

    prompt = validate_prompt(data["prompt"])
    mode, proposal = data["mode"], data["proposer"]
    require(
        mode in ("bank", *LIVE_NAMES)
        and set(data)
        == {
            "identity",
            "block",
            "prompt",
            "mode",
            "proposer",
            "recipe",
            "weights",
            "seed",
            "model",
            "shuffled",
        },
        "closed visible-only selector request required",
    )
    runtime = data["recipe"].get("base_recipe", data["recipe"])
    runtime = dict(runtime, model=dict(runtime["model"], seed=data["seed"]))
    encoder = decoder = memory = None
    snapshot = digest("registered-template-bank")
    if proposal in CONVENTIONAL:
        from .progressive_panel_worker import conventional_choice

        require(mode == "reference", "conventional incumbent outside reference mode")
        start = time.monotonic()
        result = conventional_choice(CONVENTIONAL[proposal], prompt, client)
        elapsed = time.monotonic() - start
        bank = [result["choice"]] if result["choice"] else []
        eligible_sources(bank, prompt)
        return seal(
            dict(
                bank=digest(bank),
                request=data["identity"],
                eligible=bank,
                choices={mode: result["choice"] if elapsed <= 60 else None},
                comparable={mode: elapsed <= 60},
                complete=result["complete"],
                elapsed_us=round(elapsed * 1e6),
                attempts=result["attempts"],
                seconds=60,
                seed=data["seed"],
                proposer=proposal,
            )
        )
    require(
        proposal in ("template_only", "retained_D", "trial05", "trial05_and_templates"),
        "unknown selector proposer",
    )
    if proposal != "template_only":
        method = "retained_D" if proposal == "retained_D" else "earlier_checkpoint"
        encoder, decoder, _ = load_baseline(
            method,
            data["recipe"],
            root=Path("/work"),
            ref=data["weights"],
            reference_recipe=data["recipe"],
            device="cpu",
        )
        snapshot = model_snapshot(encoder, decoder)
    start = time.monotonic()
    if encoder is not None:
        import torch

        with torch.no_grad():
            memory = encoder.forward_from_sequences(
                [[int(n) for n in prompt["values"]]], device="cpu"
            )
    deadline = start + 60
    pool = portfolio(
        decoder,
        memory,
        prompt,
        runtime,
        client,
        deadline=deadline,
        template=proposal == "template_only",
        alternate_templates=proposal == "trial05_and_templates",
        snapshot=snapshot,
        retain_selector_candidates=True,
    )
    bank = pool["selector_candidates"]["equal_budget"]
    ids = eligible_sources(bank, prompt)
    original = pool["equal_budget"]
    require(original is None or original in bank, "incumbent chose outside eligible bank")
    choices = {}
    if mode == "bank":
        choices = {
            "incumbent": original,
            "learned": choose(bank, prompt["values"], data["model"]),
            "shuffled": choose(bank, data["shuffled"], data["model"])
            if data["shuffled"] is not None
            else None,
        }
        if proposal == "trial05_and_templates":
            for parity, arm in enumerate(COMPONENT_NAMES):
                keys = {
                    key
                    for ordinal, entries in pool["proposal_sources"].items()
                    if int(ordinal) % 2 == parity
                    for key in entries
                }
                choices[arm] = choose(
                    [ids[key] for key in sorted(keys)], prompt["values"], data["model"]
                )
    else:
        choices[mode] = (
            choose(bank, prompt["values"], data["model"]) if mode == "live_learned" else original
        )
    elapsed = time.monotonic() - start
    # A live policy has to seal within its own registered 60-second allowance.
    expired = mode != "bank" and elapsed > 60
    return seal(
        dict(
            bank=digest(bank),
            request=data["identity"],
            attempts_in_order=pool["details"],
            proposal_sources=pool.get("proposal_sources", {}),
            eligible=bank,
            choices={k: None if expired else v for k, v in choices.items()},
            comparable={
                k: not expired and (k != "shuffled" or data["shuffled"] is not None)
                for k in choices
            },
            complete=pool["arm_evidence"]["equal_budget"]["complete"],
            elapsed_us=round(elapsed * 1e6),
            attempts=len(pool["details"]),
            seconds=60,
            seed=data["seed"],
            proposer=proposal,
        )
    )


def selector_score(data, client):
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
    from oeis_learn.synthesis.reference import evaluate

    truth, selected = data["truth"], data["selections"]
    required = set((*ARM_NAMES, *LIVE_NAMES))
    allowed = ({"reference","live_learned"},) if data.get("scope") == "replication" else (required, required | set(COMPONENT_NAMES))
    require(
        set(selected) in allowed
        and len(truth["indices"]) == len(truth["values"]) == 100,
        "exact selector scoring roster required",
    )
    outcomes = {}
    for arm, row in selected.items():
        envelope = row["chosen"]
        if not row["comparable"] or envelope is None:
            outcomes[arm] = "F" if row["comparable"] and row["complete"] else "U"
            continue
        eligible_sources([envelope], truth)
        body = validate_selection_envelope(envelope)["source"]
        actual = client.execute(body, truth["indices"], deadline=time.monotonic() + 4)
        independent = evaluate(body, truth["indices"], client.limits, deadline=time.monotonic() + 4)
        if actual.get("outcome") != "ok" or independent.outcome != "ok":
            outcomes[arm] = "U"
        else:
            require(
                list(independent.values) == actual["values"]
                and actual["values"][:20] == truth["values"][:20],
                "independent selector verdict or prefix differs",
            )
            outcomes[arm] = "S" if actual["values"] == truth["values"] else ("F" if row["complete"] else "U")
    return {"outcomes": outcomes}


def compare_selector(area, registration, roster, witnesses, model, native, recipes, weight_inputs, *, replication=None):
    from .progressive_round028_pilot import outcome_channels

    visible, truth = outcome_channels({"members": roster}, witnesses)
    prompts = {r["id"]: r for r in visible}
    shuffled = shuffled_values(roster, prompts, registration["seed"])
    groups = {g:i for i, g in enumerate(sorted({m["group"] for m in roster}))}
    sizes = Counter((m["stratum"], groups[m["group"]]) for m in roster)
    arms = (*SELECTOR_ARMS, *COMPONENT_NAMES) if registration["proposer"] == "trial05_and_templates" else SELECTOR_ARMS
    design = seal(dict(version="round028-selector-design/v1", registration=registration["identity"],
                       roster=digest(roster), shuffle=digest(shuffled), model=model["identity"],
                       reference=registration["reference"]["method"], arms=list(arms), seed=registration["seed"],
                       profile=dict(attempts=8, portfolio_us=60_000_000, fit_us=4_000_000, tokens=1024),
                       sizes=[dict(stratum=s, group=g, cases=n) for (s,g),n in sorted(sizes.items())],
                       **({"replication":dict(identity=replication["identity"],seconds=replication["seconds"])} if replication else {})))
    retain(area / "design.json", design)
    selected = {}
    charges = []
    searches = {}
    for member in roster:
        key = member["task"]
        choices = {}
        for mode in ("bank", *LIVE_NAMES):
            proposer = (
                registration["reference"]["method"]
                if mode == "reference"
                else registration["proposer"]
            )
            recipe = recipes[proposer]
            weights, mounts = weight_inputs[proposer]
            packet = seal(
                dict(
                    block=digest([key, mode])[7:],
                    prompt=prompts[key],
                    mode=mode,
                    proposer=proposer,
                    recipe=recipe,
                    weights=weights,
                    seed=registration["seed"],
                    model=model,
                    shuffled=shuffled[key],
                )
            )
            result = native("selector_choice", packet, seconds=600, inputs=mounts)
            require(
                result == seal(result) and result["request"] == packet["identity"]
                and result["bank"] == digest(result["eligible"]) and result["seconds"] == 60
                and result["seed"] == registration["seed"] and result["proposer"] == proposer
                and set(result["choices"]) == (set(ARM_NAMES) | (set(COMPONENT_NAMES) if len(arms) == 8 else set())
                                                if mode == "bank" else {mode}),
                "unsealed eligible bank",
            )
            eligible_sources(result["eligible"], prompts[key])
            path = area / "banks" / (packet["block"] + ".json")
            retain(path, result)
            artifact = reference(area, path)
            searches[packet["block"]] = artifact
            for arm, choice in result["choices"].items():
                require(choice is None or choice in result["eligible"], "substituted selector bank")
                choices[arm] = dict(
                    chosen=choice,
                    bank=result["bank"],
                    producer=artifact,
                    complete=result["complete"],
                    comparable=result["comparable"][arm],
                )
            charges.append(
                dict(mode=mode, wall_us=native.last_wall_us, search_us=result["elapsed_us"])
            )
        selected[key] = choices
    freeze = seal(
        dict(registration=registration["identity"], model=model["identity"], choices=selected)
    )
    retain(area / "selections.json", freeze)
    results = []
    scores = {}
    for member, row in zip(roster, truth, strict=True):
        key = member["task"]
        scored = native(
            "selector_score",
            seal(
                dict(
                    block=digest(["score", key])[7:],
                    truth=row,
                    parent=freeze["identity"],
                    selections=selected[key],
                )
            ),
            seconds=300,
        )
        charges.append(dict(mode="score", wall_us=native.last_wall_us, search_us=0))
        scored_path = area / "scores" / (digest(key)[7:] + ".json")
        retain(scored_path, seal(dict(parent=freeze["identity"], truth=digest(row), outcomes=scored["outcomes"])))
        scores[key] = reference(area, scored_path)
        results.append(dict(member=member, outcomes=scored["outcomes"]))
    retain(area / "closure.json", seal(dict(design=design["identity"], selections=freeze["identity"],
                                           searches=searches, scores=scores, charges=charges)))
    joint = defaultdict(Counter)
    for row in results:
        member = row["member"]
        joint[member["stratum"], groups[member["group"]]]["".join(row["outcomes"][a] for a in arms)] += 1
    cells = [dict(stratum=s, group=g, counts=dict(c)) for (s,g),c in sorted(joint.items())]
    counts, pairs = selector_counts(cells, arms)
    return seal(
        dict(
            version="round028-selector-report/v1", design=design, cells=cells,
            spending=selector_spending(cells, comparable_cost=True,repeat_funded=bool(replication)),
            registration=registration["identity"],
            selections=freeze["identity"],
            cases=len(roster),
            arms=counts,
            pairs=pairs,
            elapsed_us=sum(charge["wall_us"] for charge in charges),
            search_us=sum(charge["search_us"] for charge in charges),
            costs={mode:sum(c["wall_us"] for c in charges if c["mode"] == mode)
                   for mode in ("bank", *LIVE_NAMES, "score")},
            gpu_dispatches=0,
        )
    )


def selector_command(args, source, root, folder, revision, store, host_factory, audit):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, load
    from oeis_learn.generalization.progressive.training_rows import held_out_probes, verified_training_pairs, pair_seeds
    from .progressive_round028 import reopen
    from .progressive_round028_audit import ORIGINAL, Donor
    from .progressive_round028_panels import NativeRoles, prepare
    from .progressive_round028_replay import DiagnosticRole
    from .progressive_round028_support import training_package
    from .progressive_panel_sources import read_inputs
    from .progressive_research_inputs import checked_file, _json
    from .progressive_round028_pilot import research_rosters, PilotRole, prepare_training

    require(audit["decision"]["action"] == "CPU_SELECTOR", "CPU selector audit required")
    donor = Donor(root, folder, store.campaign)
    campaign = donor.saved(
        ORIGINAL + "/prepare-research/frozen/campaign.json",
        donor.policy["campaign"],
        maximum=CAMPAIGN_DOCUMENT_BYTES,
    )
    refs = {r["name"]: r for r in campaign["payload"]["references"]}
    package = training_package(donor, refs)
    probes = held_out_probes(package["layout"]) if package else set()
    pairs = verified_training_pairs(package["shards"], excluded=probes)["pairs"] if package else []
    if not pairs:
        return dict(
            mechanical_status="BLOCKED", reason="SELECTOR_TRAINING_PAIR_DEFICIT", eligible_pairs=0
        ), 2
    seed = int(digest([store.campaign["identity"], "selector"])[7:15], 16)
    cut = reopen(root, folder, store.campaign, store.state["audit"])
    trial = cut["producers"].get("trial-05-1M", {})
    proposer = "template_only"
    base_recipe = _json(checked_file(root, refs["recipe"]["artifact"]))
    recipes = dict(template_only=base_recipe, retained_D=base_recipe)

    def mount(ref):
        return dict(ref, path="inputs/weights.pt"), [
            (checked_file(root, ref), "/work/inputs/weights.pt")
        ]

    weight_inputs = dict(
        template_only=(None, []),
        retained_D=mount(refs["retained-D-evaluation-weights"]["artifact"]),
    )
    index = donor.replay_index() or {}
    for task in campaign["payload"]["tasks"]:
        config = task.get("trial") or {}
        checkpoint = index.get("training_checkpoints", {}).get(task["id"], {})
        if (
            task["kind"] == "training"
            and config.get("ordinal") == 5
            and task["token_target"] == 1_000_000
            and campaign["track"] == "synthetic_strict"
            and task["initializer"].get("mode") in ("random", "fresh_random")
            and "checkpoint" in trial
            and any(
                r["sha256"] == trial["checkpoint"]["sha256"] for r in checkpoint.get("files", [])
            )
        ):
            from oeis_learn.generalization.progressive.import_state import inspect
            from oeis_learn.generalization.progressive.trial_scopes import materialize, variants

            state, costs, _, full, original_files = inspect(root, checkpoint["source"])
            require(
                state["track"] == "synthetic_strict"
                and state["payload"]["campaign"] == donor.policy["campaign"]
                and state["payload"]["task"] == task["id"]
                and costs["milestone"]["tokens"] >= 1_000_000
                and full["states"]["weights"]["sha256"] == trial["checkpoint"]["sha256"]
                and original_files == checkpoint["files"],
                "trial proposer lineage changed",
            )
            matches = [
                p
                for p in variants(task)
                if p["initializer"]["compatibility"] == state["payload"]["compatibility"]
            ]
            if len(matches) != 1:
                continue
            proposer = "trial05_and_templates"
            recipes["trial05"] = recipes[proposer] = materialize(package, matches[0])["profile"]
            weight_inputs["trial05"] = weight_inputs[proposer] = mount(trial["checkpoint"])
            break
    for name in CONVENTIONAL:
        recipes[name] = base_recipe
        weight_inputs[name] = (None, [])
    incumbent = historical_reference(cut, trial_eligible=proposer == "trial05_and_templates")
    area = confined(folder, "private/cpu-selector-v2", exists=False)
    seconds = getattr(args, "selector_repeat_seconds", None)
    if seconds is None:
        old = area / "registration.json"
        seconds = load(confined(area,old)).get("repeat_seconds",0) if old.exists() else 0
    require(type(seconds) is int and seconds >= 0, "nonnegative prospective selector reservation required")
    reg = seal(
        dict(
            source=digest(collect_snapshot(source)),
            candidate=revision,
            audit=audit["receipt"]["report"],
            route=digest(audit["decision"]),
            seed=seed,
            calibration_limit=48,
            proposer=proposer,
            training_support=digest(pairs),
            profile=store.campaign["policy"]["profile"], multiplicity=1,
            repeat="requires_registered_reservation", final_assessment="SEALED",
            **({"repeat_seconds":seconds} if seconds else {}),
            reference=incumbent,
            weights={k: v[0] for k, v in weight_inputs.items()},
        )
    )
    retain(area / "registration.json", reg)
    supply_area = area / "supply"
    supply_area.mkdir(exist_ok=True)
    inventory = supply_area / "prepared-supply.json"
    if inventory.is_file():
        binding = load(confined(supply_area, inventory))
        require(
            binding == seal(binding) and binding["registration"] == reg["identity"],
            "selector supply changed",
        )
        read_bound_record(supply_area, binding["supply"])
        supply = binding["summary"]
    else:
        supply = prepare(
            root,
            supply_area,
            read_inputs(root, refs, include_protected=True),
            reg,
            NativeRoles(args, source, root, revision, supply_area, reg, host_factory),
        )
        retain(
            inventory,
            seal(
                dict(
                    registration=reg["identity"],
                    summary=supply,
                    supply=reference(supply_area, supply_area / "supply.json"),
                )
            ),
        )
    private = load(confined(supply_area, "supply.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
    if not supply["calibrated"] or private["allocations"]["deficits"]:
        return dict(
            mechanical_status="BLOCKED",
            reason="SELECTOR_PANEL_DEFICIT",
            deficits=private["allocations"]["deficits"],
        ), 2
    rosters = research_rosters(private["allocations"]["panels"])
    roster = rosters["development"] + rosters["development_retention"]
    replication = None
    reserved = rosters["replication"] + rosters["replication_retention"]
    if seconds:
        from oeis_learn.generalization.progressive.selector_replication import freeze_panel
        require(reserved and not {r["task"] for r in reserved}.intersection(r["task"] for r in roster),
                "disjoint registered replication roster required")
        replication = freeze_panel(reg["identity"],reserved,seconds)
        retain(area / "reserved-panel.json",replication)
    training_area = area / "training-0"
    training_area.mkdir(exist_ok=True)
    cpu = PilotRole(args, source, root, revision, training_area, reg, host_factory)
    training = prepare_training(root, training_area, reg, cpu, refs, private, supply_area)
    model = rank_training_stream(training, pair_seeds(reg["identity"], 0)["training_families"])
    retain(
        area / "freeze.json",
        seal(dict(registration=reg["identity"], panels=digest(rosters), model=model)),
    )
    native = DiagnosticRole(args, source, root, revision, area, reg, host_factory)
    result = compare_selector(
        area, reg, roster, private["witnesses"], model, native, recipes, weight_inputs, replication=replication
    )
    retain(area / "result.json", result)
    from io import BytesIO
    from oeis_learn.generalization.progressive.contracts import _atomic_write
    from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode
    packed = archive_bytes({"selector.json":encode(result)})
    receipt = verify_selector(BytesIO(packed))
    share = confined(folder, "share/round028-selector.zip", exists=False)
    if share.exists():
        require(share.read_bytes() == packed, "existing selector publication differs")
    else:
        _atomic_write(share, packed)
    if receipt["next_action"] == "RUN_RESERVED_REPLICATION":
        from .progressive_selector_replication import run_repeat
        def runner(work, *, training=False):
            cls = PilotRole if training else DiagnosticRole
            role = cls(args,source,root,revision,work,reg,host_factory)
            role.repeatable_preparation = frozenset()
            return role
        return run_repeat(area,folder,reg,replication,reserved,private["witnesses"],model,runner,
                lambda work,role: prepare_training(root,work,reg,role,refs,private,supply_area,pair=1),
                recipes,weight_inputs,packed)
    return dict(
        mechanical_status="COMPLETE", scientific_status="SELECTOR_EXPLORATORY", result=result,
        receipt=receipt, archive=str(share.relative_to(folder)), next_action=receipt["next_action"]
    ), 0


def verify_private_selector(area, share):
    """Reopen choices, ordered producers and the frozen roster without dispatching a role."""
    from oeis_learn.generalization.progressive.contracts import load, CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
    from oeis_learn.synthesis.reference import evaluate
    from .progressive_round028_pilot import research_rosters, outcome_channels
    def record(name):
        item = load(confined(area, name), maximum=CAMPAIGN_DOCUMENT_BYTES)
        require(item == seal(item), "changed private selector record")
        return item

    public = verify_selector(share)
    result, closure, freeze = (record(n) for n in ("result.json", "closure.json", "freeze.json"))
    choices, design, reg = (record(n) for n in ("selections.json", "design.json", "registration.json"))
    supply_cut = record("supply/prepared-supply.json")
    supply = read_bound_record(area / "supply", supply_cut["supply"])
    panels = research_rosters(supply["allocations"]["panels"])
    roster = panels["development"] + panels["development_retention"]
    require(freeze["panels"] == digest(panels) and design["roster"] == digest(roster)
            and choices["registration"] == freeze["registration"] == design["registration"] == reg["identity"]
            and freeze["model"]["identity"] == choices["model"] == design["model"]
            and choices["identity"] == result["selections"] == closure["selections"]
            and design["identity"] == closure["design"] == public["design"]
            and public["report"] == result["identity"], "selector private/public parent differs")
    visible, truths = outcome_channels({"members":roster}, supply["witnesses"])
    shuffled = shuffled_values(roster, {r["id"]:r for r in visible}, reg["seed"])
    require(design["shuffle"] == digest(shuffled), "selector conditioning donor differs")
    groups = {g:i for i,g in enumerate(sorted({m["group"] for m in roster}))}
    joint = defaultdict(Counter)
    for member, prompt, truth in zip(roster, visible, truths, strict=True):
        key = member["task"]
        sealed = choices["choices"][key]
        score = read_bound_record(area, closure["scores"][key])
        require(score["parent"] == choices["identity"] and score["truth"] == digest(truth),
                "selector scorer parent differs")
        for mode in ("bank", *LIVE_NAMES):
            block = digest([key, mode])[7:]
            producer = read_bound_record(area, closure["searches"][block])
            bank = producer["eligible"]
            require(producer["bank"] == digest(bank) and producer["seed"] == reg["seed"], "selector bank changed")
            eligible_sources(bank, prompt)
            for arm, chosen in producer["choices"].items():
                row = sealed[arm]
                require(row == dict(chosen=chosen, bank=producer["bank"], producer=closure["searches"][block],
                                    comparable=producer["comparable"][arm], complete=producer["complete"])
                        and (chosen is None or chosen in bank), "selector choice differs from its bank")
                if arm in ("learned", "shuffled", "live_learned") and row["comparable"]:
                    values = shuffled[key] if arm == "shuffled" else prompt["values"]
                    require(chosen == choose(bank, values, freeze["model"]), "learned choice differs")
                verdict = score["outcomes"][arm]
                if chosen is None or not row["comparable"]:
                    expected = "F" if row["comparable"] and row["complete"] else "U"
                    require(verdict == expected, "empty or late selection differs")
                elif verdict != "U":
                    source = validate_selection_envelope(chosen)["source"]
                    execution = evaluate(source, truth["indices"])
                    require(execution.outcome == "ok" and
                            verdict == ("S" if list(execution.values) == truth["values"] else "F")
                            and (verdict == "S" or row["complete"]), "private continuation differs")
        joint[member["stratum"], groups[member["group"]]]["".join(score["outcomes"][a] for a in design["arms"])] += 1
    rebuilt = [dict(stratum=s, group=g, counts=dict(c)) for (s,g),c in sorted(joint.items())]
    by_mode = Counter()
    for charge in closure["charges"]:
        by_mode[charge["mode"]] += charge["wall_us"]
    require(rebuilt == result["cells"] and by_mode == result["costs"]
            and sum(c["search_us"] for c in closure["charges"]) == result["search_us"], "selector reduction differs")
    if (area / "reserved-panel.json").is_file():
        from oeis_learn.generalization.progressive.selector_replication import freeze_panel, verify_replication
        from .progressive_selector_replication import SHARE, reopen as reopen_replication
        registered = record("reserved-panel.json")
        reserved = panels["replication"] + panels["replication_retention"]
        require(registered == freeze_panel(reg["identity"],reserved,reg["repeat_seconds"])
                and registered["identity"] == public["replication"], "selector repeat registration differs")
        path = confined(share.parent.parent, SHARE, exists=False)
        if path.is_file():
            rep = record("replication/result.json")
            models,cells = reopen_replication(area / "replication",registered,reserved,supply["witnesses"])
            from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
            budget=BudgetJournal(area / "replication/budget",registered["identity"],
                        dict(dict.fromkeys(KEYS,0),elapsed_us=reg["repeat_seconds"]*1_000_000)).snapshot()
            public=verify_replication(path)
            require(cells==rep["cells"] and models==rep["models"] and budget["consumed"]["elapsed_us"]==rep["charged_us"]
                    and not budget["pending"] and public["report"]==rep["identity"], "selector replication replay differs")
    return dict(private_replay="VERIFIED", public_recomputation=public)


def rank_training_stream(training, seed):
    stream = training["stream"]
    require(stream == seal(stream) and stream["seeds"]["training_families"] == seed,
            "selector training seed differs")
    rows = {r["draw_identity"]:r for r in training["rows"]}
    examples = [dict(case=p["positive_row"], values=rows[p["positive_row"]]["values"][:20],
                     positive=rows[p["positive_row"]]["source"], negative=rows[p["negative_row"]]["source"])
                for p in stream["pairs"]]
    return fit_ranker(examples, seed=seed, stream=stream["identity"])
