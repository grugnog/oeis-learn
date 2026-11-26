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
)
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require

ARM_NAMES = ("incumbent", "learned", "shuffled")
LIVE_NAMES = ("live_incumbent", "live_learned", "reference")
COMPONENT_NAMES = ("model_component", "template_component")
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
        lower = upper = Fraction()
        unknown = 0
        for counts in families.values():
            total = sum(counts.values())
            missing = total - counts["success"] - counts["failure"]
            unknown += missing
            lower += Fraction(counts["success"], total)
            upper += Fraction(counts["success"] + missing, total)
        bounds[name] = (lower / len(families), upper / len(families), unknown)
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
    require(
        set(selected) in (required, required | set(COMPONENT_NAMES))
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
            outcomes[arm] = "S" if actual["values"] == truth["values"] else "F"
    return {"outcomes": outcomes}


def compare_selector(area, registration, roster, witnesses, model, native, recipes, weight_inputs):
    from .progressive_round028_pilot import outcome_channels

    visible, truth = outcome_channels({"members": roster}, witnesses)
    prompts = {r["id"]: r for r in visible}
    shuffled = shuffled_values(roster, prompts, registration["seed"])
    selected = {}
    charges = []
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
                result == seal(result) and result["bank"] == digest(result["eligible"]),
                "unsealed eligible bank",
            )
            eligible_sources(result["eligible"], prompts[key])
            for arm, choice in result["choices"].items():
                require(choice is None or choice in result["eligible"], "substituted selector bank")
                choices[arm] = dict(
                    chosen=choice,
                    bank=result["bank"],
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
        results.append(dict(member=member, outcomes=scored["outcomes"]))
    counts = {}
    for name in results[0]["outcomes"]:
        counts[name] = dict(Counter(r["outcomes"][name] for r in results))
    pairs = {}
    for a, b in (
        ("learned", "incumbent"),
        ("learned", "shuffled"),
        ("live_learned", "live_incumbent"),
        ("live_learned", "reference"),
    ):
        grouped = {}
        for r in results:
            group = (r["member"]["stratum"], r["member"]["group"])
            grouped.setdefault(group, Counter())[r["outcomes"][a] + r["outcomes"][b]] += 1
        pairs[a + "_minus_" + b] = [
            dict(stratum=s, counts=dict(c)) for (s, _), c in sorted(grouped.items())
        ]
    return seal(
        dict(
            registration=registration["identity"],
            selections=freeze["identity"],
            cases=len(roster),
            arms=counts,
            pairs=pairs,
            elapsed_us=sum(charge["wall_us"] for charge in charges),
            search_us=sum(charge["search_us"] for charge in charges),
            gpu_dispatches=0,
        )
    )


def selector_command(args, source, root, folder, revision, store, host_factory, audit):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, load
    from oeis_learn.generalization.progressive.shared_support import validated_rows
    from oeis_learn.generalization.progressive.training_rows import verified_training_pairs
    from .progressive_round028 import reopen
    from .progressive_round028_audit import ORIGINAL, Donor
    from .progressive_round028_panels import NativeRoles, prepare
    from .progressive_round028_replay import DiagnosticRole
    from .progressive_round028_support import training_package
    from .progressive_panel_sources import read_inputs
    from .progressive_research_inputs import checked_file, _json
    from .progressive_round028_pilot import research_rosters

    require(audit["decision"]["action"] == "CPU_SELECTOR", "CPU selector audit required")
    donor = Donor(root, folder, store.campaign)
    campaign = donor.saved(
        ORIGINAL + "/prepare-research/frozen/campaign.json",
        donor.policy["campaign"],
        maximum=CAMPAIGN_DOCUMENT_BYTES,
    )
    refs = {r["name"]: r for r in campaign["payload"]["references"]}
    package = training_package(donor, refs)
    probes = [p["id"] for p in (package or {}).get("layout", {}).get("probes", {}).values()]
    pairs = verified_training_pairs(package["shards"], excluded=probes)["pairs"] if package else []
    if not pairs:
        return dict(
            mechanical_status="BLOCKED", reason="SELECTOR_TRAINING_PAIR_DEFICIT", eligible_pairs=0
        ), 2
    rows = validated_rows(package["shards"], native_verified=True)
    examples = [
        dict(
            case=p["positive_row"],
            values=rows[p["positive_row"]]["values"][:20],
            positive=rows[p["positive_row"]]["source"],
            negative=rows[p["negative_row"]]["source"],
        )
        for p in pairs
    ]
    model = fit_ranker(examples)
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
    reg = seal(
        dict(
            source=digest(collect_snapshot(source)),
            candidate=revision,
            audit=audit["receipt"]["report"],
            route=digest(audit["decision"]),
            seed=int(digest([store.campaign["identity"], "selector"])[7:15], 16),
            calibration_limit=48,
            proposer=proposer,
            training=model["identity"],
            reference=incumbent,
            weights={k: v[0] for k, v in weight_inputs.items()},
        )
    )
    area = confined(folder, "private/cpu-selector", exists=False)
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
    retain(
        area / "freeze.json",
        seal(dict(registration=reg["identity"], panels=digest(rosters), model=model)),
    )
    native = DiagnosticRole(args, source, root, revision, area, reg, host_factory)
    result = compare_selector(
        area, reg, roster, private["witnesses"], model, native, recipes, weight_inputs
    )
    retain(area / "result.json", result)
    return dict(
        mechanical_status="COMPLETE", scientific_status="SELECTOR_EXPLORATORY", result=result
    ), 0
