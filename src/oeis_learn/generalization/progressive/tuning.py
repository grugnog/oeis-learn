"""Serial pinned Optuna proposals and prospective screening feasibility."""

from statistics import median

from .contracts import digest
from .trials import integer, scalar, should_prune

SPACE = {
    "lr_multiplier": [0.3, 1, 3],
    "schedule": ["constant", "warmup_cosine"],
    "effective_batch": [16, 32, 64],
    "weight_decay": [0, 0.01, 0.1],
    "dropout": [0, 0.1, 0.2],
}
BASELINE = {
    "lr_multiplier": 1,
    "schedule": "constant",
    "effective_batch": 16,
    "weight_decay": 0.01,
    "dropout": 0,
}
CONTROL = dict(BASELINE, lr_multiplier=0.3)
SHAPES = ("reference", "smaller_width", "shallower", "larger")


def propose(journal, ordinal, *, selected_learning=None):
    """Rebuild ordered history; reuse a sealed proposal before asking again."""
    import optuna
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState, create_trial

    if optuna.__version__ != "4.5.0":
        raise ValueError("pinned Optuna 4.5.0 required")
    integer(ordinal)
    journal.read()
    if str(ordinal) in journal.trials:
        return journal.proposal(ordinal)
    if ordinal >= 16 or ordinal != len(journal.trials):
        raise ValueError("finite ordered trial roster required")
    if any(
        s["status"] not in {"COMPLETE", "PRUNED", "FAILED", "PARTIAL", "INFEASIBLE"}
        for s in journal.trials.values()
    ):
        raise ValueError("serial completed history required")
    seed = int(digest([journal.binding["study"], ordinal]).split(":")[1][:8], 16)
    if ordinal >= 12:
        if (
            not isinstance(selected_learning, dict)
            or set(selected_learning) != set(SPACE)
            or any(selected_learning[k] not in SPACE[k] for k in SPACE)
        ):
            raise ValueError("locked learning selection required before sizing")
        params = dict(selected_learning, shape=SHAPES[ordinal - 12])
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4 if ordinal < 4 else 0),
        )
        distributions = {k: CategoricalDistribution(v) for k, v in SPACE.items()}
        for s in journal.trials.values():
            result = s["result"]
            state = (
                TrialState.COMPLETE
                if s["status"] == "COMPLETE"
                else TrialState.PRUNED
                if s["status"] == "PRUNED"
                else TrialState.FAIL
            )
            study.add_trial(
                create_trial(
                    params=s["proposal"]["parameters"],
                    distributions=distributions,
                    value=result["value"] if state == TrialState.COMPLETE else None,
                    state=state,
                )
            )
        if ordinal in (0, 1):
            study.enqueue_trial(BASELINE if ordinal == 0 else CONTROL)
        trial = study.ask()
        params = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
    return journal.propose(ordinal, parameters=params, seed=seed)


def validate_screening_panel(rows, *, grouping="paired-16/v1"):
    if not isinstance(rows, list) or len(rows) != 128 or len({r["id"] for r in rows}) != 128:
        raise ValueError("fixed 128-task development subset required")
    if grouping not in {"paired-16/v1", "balanced-at-most-two/v2"}:
        raise ValueError("registered screening group policy required")
    for stratum in ("parameter", "composition", "unseen_structure", "oeis"):
        groups = {}
        for row in rows:
            if row["role"] != "development":
                raise ValueError("development-only screening panel")
            if row["stratum"] == stratum:
                groups[row["group"]] = groups.get(row["group"], 0) + 1
        valid=(len(groups)==16 and set(groups.values())=={2}) if grouping=="paired-16/v1" else (sum(groups.values())==32 and len(groups)>=16 and max(groups.values(),default=0)<=2)
        if not valid:
            raise ValueError("two tasks from each of sixteen groups per stratum required")
    return digest(rows)


def preflight(*, startup_us, evaluation_us, training_us, finalization_us, cap_us=7200000000, operational_revision=False):
    for n in (startup_us, training_us, finalization_us, cap_us):
        integer(n, 1)
    if len(evaluation_us) != 3:
        raise ValueError("three evaluations must be reserved")
    for n in evaluation_us:
        integer(n, 1)
    required = startup_us + sum(evaluation_us) + training_us + finalization_us
    if cap_us > (96*3600*1000000 if operational_revision else 7200000000) or required > cap_us:
        raise ValueError("screening startup/training/evaluations/finalization cannot fit")
    return required


def pruning(*, ordinal, value, coordinate, references):
    integer(ordinal)
    if ordinal >= 16:
        raise ValueError("trial roster exhausted")
    return should_prune(
        value=value,
        coordinate=coordinate,
        references=references,
        control=ordinal in (0, 1) or ordinal >= 12,
    )


def ranking_uncertainty(rows):
    values = [scalar(r["value"]) for r in rows if r["value"] is not None]
    return {
        "scope": "development ranking only",
        "seed_population_interval": None,
        "observed_range": [min(values), max(values)] if values else None,
        "median": median(values) if values else None,
    }


def propose_request(request, output):
    from .trials import TrialJournal

    if not isinstance(request, dict) or set(request) != {
        "study",
        "ordinal",
        "history",
        "selected_learning",
    }:
        raise ValueError("closed proposal request required")
    integer(request["ordinal"])
    if (
        not isinstance(request["history"], list)
        or len(request["history"]) != request["ordinal"]
        or len(request["history"]) > 15
    ):
        raise ValueError("bounded ordered history required")
    journal = TrialJournal(output, study=request["study"])
    for ordinal, entry in enumerate(request["history"]):
        if set(entry) != {"proposal", "result"} or entry["result"]["role"] != "development":
            raise ValueError("development proposal/result history only")
        proposal = entry["proposal"]
        actual = journal.propose(ordinal, parameters=proposal["parameters"], seed=proposal["seed"])
        if actual != proposal:
            raise ValueError("sealed proposal history changed")
        journal.read()
        if journal.trials[str(ordinal)]["result"] is None:
            journal.dispatch(ordinal)
            journal.tell(ordinal, entry["result"])
        elif journal.trials[str(ordinal)]["result"] != entry["result"]:
            raise ValueError("history result changed")
    return propose(journal, request["ordinal"], selected_learning=request["selected_learning"])


def propose_verified_history(request):
    """Installed sampler over host-verified scalar feedback; no truth/tensors."""
    from .contracts import seal
    if set(request)!={'version','study','ordinal','history','selected_learning'} or request['version']!='progressive-verified-proposal/v1':
        raise ValueError('closed verified development history required')
    if len(request['history'])!=request['ordinal']:raise ValueError('ordered finite feedback required')
    class History:
        binding={'study':request['study']}
        trials={str(i):entry for i,entry in enumerate(request['history'])}
        def read(self):pass
        def propose(self,ordinal,*,parameters,seed):
            return seal(dict(study=request['study'],ordinal=ordinal,parameters=parameters,seed=seed))
    for entry in request['history']:
        if set(entry)!={'proposal','status','result'} or entry['status'] not in {'COMPLETE','PRUNED','FAILED','PARTIAL','INFEASIBLE'} or entry['result']['role']!='development':
            raise ValueError('verified terminal development history only')
    return propose(History(),request['ordinal'],selected_learning=request['selected_learning'])
