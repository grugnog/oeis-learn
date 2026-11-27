"""A fixed CPU reservation and two independently trained selectors on one frozen panel."""

import io
from pathlib import Path
import time

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    _atomic_write,
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes
from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode
from oeis_learn.generalization.progressive.selector_replication import (
    formal_result,
    freeze_panel,
    joint_cells,
    verify_replication,
)
from oeis_learn.generalization.progressive.program_search_selector import choose, verify_selector
from oeis_learn.generalization.progressive.training_rows import pair_seeds
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require
from .progressive_round028_pilot import outcome_channels
from .progressive_round028_selector import eligible_sources, rank_training_stream

ARMS = ("reference", "live_learned")
SHARE = "share/round028-selector-replication.zip"


class InsufficientRepeat(Exception):
    def __init__(self, deficit):
        self.deficit = deficit


def funded_fit(area, journal, training, seed):
    """Reserve before training and recover a sealed fit without spending on it twice."""
    import signal

    packet = seal(dict(stream=training["stream"]["identity"], seed=seed))
    start = area / "ranker-start.json"
    cut = area / "ranker-result.json"
    if not start.exists():
        available = journal.snapshot()["remaining"]["elapsed_us"]
        if available <= 0:
            raise InsufficientRepeat(1)
        retain(
            start,
            seal(
                dict(
                    request=packet["identity"],
                    available=available,
                    boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    start=time.monotonic_ns() // 1000,
                )
            ),
        )
    grant = load(start)
    require(
        grant == seal(grant) and grant["request"] == packet["identity"],
        "selector training grant differs",
    )
    reserved = journal.reserve(
        packet["identity"],
        "ranker",
        0,
        dict(dict.fromkeys(KEYS, 0), elapsed_us=grant["available"]),
        boot=grant["boot"],
        monotonic_us=grant["start"],
    )
    if not cut.exists():
        if reserved.get("reserved"):
            raise InsufficientRepeat(grant["available"])

        def expire(*_):
            raise InsufficientRepeat(1)

        previous = signal.signal(signal.SIGALRM, expire)
        began = time.monotonic_ns()
        try:
            signal.setitimer(signal.ITIMER_REAL, grant["available"] / 1e6)
            model = rank_training_stream(training, seed)
            elapsed = (time.monotonic_ns() - began) // 1000
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        retain(cut, seal(dict(request=packet["identity"], model=model, elapsed_us=elapsed)))
    result = load(cut, maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(
        result == seal(result)
        and result["request"] == packet["identity"]
        and result["model"] == seal(result["model"]),
        "cached selector fit differs",
    )
    elapsed = result["elapsed_us"]
    require(
        type(elapsed) is int and 0 <= elapsed <= grant["available"],
        "selector training exceeded its reservation",
    )
    journal.commit(
        reserved["key"],
        boot=grant["boot"],
        monotonic_us=grant["start"] + elapsed,
        termination_proven=True,
    )
    return result["model"]


class FundedRole:
    """Only a verified original attempt can release the unused portion of its grant."""

    def __init__(self, runner, journal, area, pair):
        self.runner, self.journal, self.area, self.pair = runner, journal, area, pair

    def __getattr__(self, field):
        return getattr(self.runner, field)

    def __call__(self, role, data, *, seconds, inputs=()):
        key = digest([self.pair, role, data["identity"]])[7:]
        grant = self.area / "grants" / (key + ".json")
        reserved = seconds * 1_000_000
        if not grant.exists():
            available = self.journal.snapshot()["remaining"]["elapsed_us"]
            if reserved > available:
                raise InsufficientRepeat(reserved - available)
            retain(
                grant,
                seal(
                    dict(
                        request=data["identity"],
                        seconds=seconds,
                        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        start_us=time.monotonic_ns() // 1000,
                    )
                ),
            )
        original = load(grant)
        require(
            original == seal(original)
            and original["request"] == data["identity"]
            and original["seconds"] == seconds,
            "selector grant changed",
        )
        claim = self.journal.reserve(
            key,
            role,
            0,
            dict(dict.fromkeys(KEYS, 0), elapsed_us=reserved),
            boot=original["boot"],
            monotonic_us=original["start_us"],
        )
        value = self.runner(role, data, seconds=seconds, inputs=inputs)
        elapsed = self.runner.last_wall_us
        require(
            type(elapsed) is int and elapsed >= 0,
            "selector role lacks its original measured interval",
        )
        self.journal.commit(
            claim["key"],
            boot=original["boot"],
            monotonic_us=original["start_us"] + min(reserved, elapsed),
            termination_proven=True,
        )
        if elapsed > reserved:
            raise InsufficientRepeat(elapsed - reserved)
        return value


def reopen(area, design, roster, witnesses):
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
    from oeis_learn.synthesis.reference import evaluate

    visible, truths = outcome_channels({"members": roster}, witnesses)
    models, scores = [], []
    for pair in (0, 1):
        work = area / str(pair)
        model = load(confined(work, "model.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
        bank = load(confined(work, "choices.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
        cut = load(confined(work, "scores.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
        require(
            model == seal(model)
            and bank == seal(bank)
            and cut == seal(cut)
            and bank["panel"] == design["identity"]
            and bank["model"] == model["identity"]
            and cut["choices"] == bank["identity"],
            "replication private parents differ",
        )
        models.append(
            dict(
                seed=model["seed"],
                model=model["identity"],
                training=model["training"],
                stream=model["stream"],
            )
        )
        outcomes = {}
        for member, prompt, truth in zip(roster, visible, truths, strict=True):
            key = member["task"]
            row = read_bound_record(work, cut["scores"][key])
            require(
                row["parent"] == bank["identity"]
                and row["truth"] == digest(truth)
                and set(row["outcomes"]) == set(ARMS),
                "replication scorer parent differs",
            )
            for arm in ARMS:
                produced = read_bound_record(work, bank["choices"][key][arm])
                allowed = produced["eligible"]
                require(
                    produced["seed"] == pair_seeds(design["registration"], pair)["initialization"]
                    and produced["bank"] == digest(allowed)
                    and set(produced["choices"]) == {arm}
                    and produced["seconds"] == 60,
                    "replication portfolio differs",
                )
                eligible_sources(allowed, prompt)
                choice = produced["choices"][arm]
                complete, comparable = produced["complete"], produced["comparable"][arm]
                require(
                    choice is None or choice in allowed, "replication choice outside eligible bank"
                )
                if arm == "live_learned" and comparable:
                    require(
                        choice == choose(allowed, prompt["values"], model),
                        "replication ranking differs",
                    )
                value = row["outcomes"][arm]
                if not comparable or choice is None:
                    require(
                        value == ("F" if comparable and complete else "U"),
                        "replication missing choice differs",
                    )
                elif value != "U":
                    actual = evaluate(
                        validate_selection_envelope(choice)["source"], truth["indices"]
                    )
                    require(
                        actual.outcome == "ok"
                        and value == ("S" if list(actual.values) == truth["values"] else "F")
                        and (value == "S" or complete),
                        "replication exact continuation differs",
                    )
            outcomes[key] = row["outcomes"]
        scores.append(outcomes)
    return models, joint_cells(roster, scores)


def measure_pair(area, pair, reg, design, roster, witnesses, model, native, recipes, weight_inputs):
    visible, truths = outcome_channels({"members": roster}, witnesses)
    seed = pair_seeds(reg["identity"], pair)["initialization"]
    selected, refs = {}, {}
    retain(area / "model.json", model)
    for prompt in visible:
        key = prompt["id"]
        selected[key], refs[key] = {}, {}
        for mode in ARMS:
            proposer = reg["reference"]["method"] if mode == "reference" else reg["proposer"]
            weights, mounts = weight_inputs[proposer]
            packet = seal(
                dict(
                    block=digest([key, mode])[7:],
                    prompt=prompt,
                    mode=mode,
                    proposer=proposer,
                    recipe=recipes[proposer],
                    weights=weights,
                    seed=seed,
                    model=model,
                    shuffled=None,
                )
            )
            output = native("selector_choice", packet, seconds=600, inputs=mounts)
            require(
                output == seal(output)
                and output["request"] == packet["identity"]
                and output["proposer"] == proposer
                and output["seed"] == seed
                and set(output["choices"]) == {mode},
                "replication choice request differs",
            )
            path = area / "banks" / (packet["block"] + ".json")
            retain(path, output)
            refs[key][mode] = reference(area, path)
            selected[key][mode] = dict(
                chosen=output["choices"][mode],
                complete=output["complete"],
                comparable=output["comparable"][mode],
            )
    freeze = seal(dict(panel=design["identity"], model=model["identity"], choices=refs))
    retain(area / "choices.json", freeze)
    scores = {}
    for truth in truths:
        key = truth["id"]
        packet = seal(
            dict(
                block=digest(["score", key])[7:],
                truth=truth,
                parent=freeze["identity"],
                scope="replication",
                selections=selected[key],
            )
        )
        result = native("selector_score", packet, seconds=300)
        path = area / "scores" / (digest(key)[7:] + ".json")
        retain(
            path,
            seal(dict(parent=freeze["identity"], truth=digest(truth), outcomes=result["outcomes"])),
        )
        scores[key] = reference(area, path)
    retain(area / "scores.json", seal(dict(choices=freeze["identity"], scores=scores)))


def run_repeat(
    area,
    folder,
    reg,
    design,
    roster,
    witnesses,
    first_model,
    runner,
    prepare,
    recipes,
    weight_inputs,
    development,
):
    rep = area / "replication"
    require(
        design == freeze_panel(reg["identity"], roster, reg["repeat_seconds"]),
        "reserved selector panel differs",
    )
    dev = verify_selector(io.BytesIO(development))
    require(
        dev["replication"] == design["identity"]
        and dev["next_action"] == "RUN_RESERVED_REPLICATION",
        "selector spending gate is closed",
    )
    journal = BudgetJournal(
        rep / "budget",
        design["identity"],
        dict(dict.fromkeys(KEYS, 0), elapsed_us=design["seconds"] * 1_000_000),
    )
    try:
        train = area / "training-1"
        train.mkdir(exist_ok=True)
        training = prepare(train, FundedRole(runner(train, training=True), journal, rep, 1))
        second = funded_fit(train, journal, training, design["seeds"][1])
        require(
            first_model["seed"] == design["seeds"][0]
            and first_model["stream"] != second["stream"]
            and first_model["training"] != second["training"],
            "distinct registered selector training streams required",
        )
        for pair, model in enumerate((first_model, second)):
            work = rep / str(pair)
            work.mkdir(exist_ok=True)
            measure_pair(
                work,
                pair,
                reg,
                design,
                roster,
                witnesses,
                model,
                FundedRole(runner(work), journal, rep, pair),
                recipes,
                weight_inputs,
            )
    except InsufficientRepeat as exc:
        return (
            dict(
                mechanical_status="BLOCKED",
                reason="SELECTOR_REPLICATION_CPU_DEFICIT",
                deficit_us=exc.deficit,
                available_us=journal.snapshot()["remaining"]["elapsed_us"],
                assessment="SEALED",
            ),
            2,
        )
    models, cells = reopen(rep, design, roster, witnesses)
    result = formal_result(design, models, cells)
    record = seal(
        dict(
            version="round028-selector-replication/v1",
            development=dev["report"],
            design=design,
            models=models,
            cells=cells,
            charged_us=journal.snapshot()["consumed"]["elapsed_us"],
            result=result,
        )
    )
    retain(rep / "result.json", record)
    packed = archive_bytes({"replication.json": encode(record), "development.zip": development})
    receipt = verify_replication(io.BytesIO(packed))
    path = confined(folder, SHARE, exists=False)
    if path.exists():
        require(path.read_bytes() == packed, "published selector replication differs")
    else:
        _atomic_write(path, packed)
    return (
        dict(
            mechanical_status="COMPLETE",
            scientific_status=receipt["verdict"],
            receipt=receipt,
            archive=SHARE,
            next_action=receipt["next_action"],
        ),
        0,
    )
