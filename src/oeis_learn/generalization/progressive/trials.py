"""Bounded durable trial state machine; sampler feedback is development-only."""

import copy
import math
import re
from contextlib import contextmanager
from pathlib import Path
from statistics import median

from oeis_learn.pipeline_validation.identity import seal

from .contracts import atomic_write, digest, load, record

TERMINAL = {"COMPLETE", "PRUNED", "FAILED", "PARTIAL", "INFEASIBLE"}


def integer(n, minimum=0):
    if type(n) is not int or not minimum <= n <= 2**63 - 1:
        raise ValueError("bounded integer required")
    return n


def scalar(n):
    if type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1:
        raise ValueError("finite capability fraction required")
    return n


def should_prune(*, value, coordinate, references, control=False):
    scalar(value)
    integer(coordinate)
    if type(control) is not bool:
        raise ValueError("explicit control flag required")
    eligible = [
        scalar(r["value"])
        for r in references
        if r["coordinate"] == coordinate
        and r["status"] == "COMPLETE"
        and r["role"] == "development"
    ]
    return (
        not control
        and coordinate in (1000000, 2000000)
        and len(eligible) >= 5
        and value < median(eligible)
    )


class TrialJournal:
    def __init__(self, root, *, study, token_cap=4000000, elapsed_cap_us=7200000000,provenance_root=None, readonly=False):
        if not isinstance(study, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", study) is None:
            raise ValueError("sealed study identity required")
        self.root = Path(root)
        self.readonly = readonly
        self.provenance_root=Path(provenance_root) if provenance_root is not None else self.root
        if self.root.is_symlink():
            raise ValueError("journal symlink")
        if not readonly:
            self.root.mkdir(parents=True, exist_ok=True)
        self.binding = seal(
            {
                "version": "progressive-trial-journal/v1",
                "study": study,
                "token_cap": integer(token_cap, 1),
                "elapsed_cap_us": integer(elapsed_cap_us, 1),
            }
        )
        with self.lock():
            p = self.root / "binding.json"
            if readonly or p.exists():
                if load(p) != self.binding:
                    raise ValueError("trial allowance cannot change")
            else:
                atomic_write(p, self.binding)
            self.read()

    def observe_pilot(self, observation):
        """Persist an exploratory look without giving the 013 optimizer terminal feedback."""
        targets = (0, 1_000_000, 4_000_000)
        target = observation["target"]
        if (observation.get("version") != "round028-pilot-observation/v1" or seal(observation) != observation
                or observation["registration"] != self.binding["study"] or observation["pair"] != 0 or target not in targets):
            raise ValueError("registered pilot observation required")
        with self.lock():
            previous = self.binding["identity"]
            for look in targets[:targets.index(target)]:
                record = load(self.root / f"pilot-{look}.json")
                if record != seal(record) or record["previous"] != previous or record["observation"]["target"] != look:
                    raise ValueError("pilot observation chain differs")
                previous = record["identity"]
            result = seal(dict(previous=previous, observation=observation))
            path = self.root / f"pilot-{target}.json"
            if path.exists():
                if load(path) != result:
                    raise ValueError("conflicting pilot observation")
            else:
                atomic_write(path, result)
            return result["identity"]

    def register_velocity(self, registration):
        from .campaign import validate_velocity_registration
        from .velocity_trials import recipe_bindings
        from oeis_learn.pipeline_validation.preparation_recovery import retain

        reg = validate_velocity_registration(registration)
        if reg["identity"] != self.binding["study"]:
            raise ValueError("different study journal")
        with self.lock():
            retain(self.root / "velocity-registration.json", reg)
        bindings = recipe_bindings(reg["identity"], initialization=reg["initialization"],
                                   stream_seed=reg["stream_seed"], ordinary=reg["ordinary"], broad=reg["broad"],
                                   training_start=reg.get("training_start"))
        methods = list(bindings) if reg["scope"] == "exploration" else ["00", reg["finalist"]]
        for ordinal, method in enumerate(methods):
            binding = bindings[method]
            row = binding["recipe"]
            parameters = dict(lr_multiplier=1, schedule="constant", effective_batch=row["batch"],
                              weight_decay=row["decay"], dropout=0)
            proposal = self.propose(ordinal, parameters=parameters, seed=reg["initialization"])
            with self.lock():
                retain(self.root / ("trial-" + method + ".json"), seal(dict(
                    version="round029-registered-trial/v1", binding=binding, proposal=proposal["identity"],
                    ordinal=ordinal)))

    def velocity_trial(self, method):
        from .campaign import velocity_policy

        if method not in {row["id"] for row in velocity_policy()["recipes"]}:
            raise ValueError("unregistered recipe")
        with self.lock():
            entry = load(self.root / ("trial-" + method + ".json"))
            proposal = self.proposal(entry["ordinal"])
            binding = entry["binding"]
            if (entry != seal(entry) or binding != seal(binding) or binding["registration"] != self.binding["study"]
                    or binding["recipe"]["id"] != method or entry["proposal"] != proposal["identity"]):
                raise ValueError("registered trial differs")
            return entry

    def observe_velocity(self, registration, look, *, evidence=None, reason=None):
        """An interim measurement or a missed look always keeps its frozen denominator."""
        from .contracts import _check, schema
        from .graph import velocity_study_looks
        from oeis_learn.pipeline_validation.preparation_recovery import retain
        from oeis_learn.pipeline_validation.identity import confined

        with self.lock():
            reg = load(self.root / "velocity-registration.json")
            if reg != registration or reg["identity"] != self.binding["study"]:
                raise ValueError("observation must use the original frozen study")
            series = velocity_study_looks(reg).get(look["method"], {}).get(look["profile"], [])
            series = [entry for entry in series if entry["seed"] == look["seed"]]
            if look not in series:
                raise ValueError("observation outside frozen roster or profile")
            detail = None
            if isinstance(evidence, dict):
                from .velocity_metrics import measured_look

                detail = measured_look(evidence["probe"], evidence["scores"],
                                       cut=evidence["cut"], work=evidence["work"])
                if detail != evidence or detail["probe"]["look"] != look or detail["probe"]["mode"] != "true":
                    raise ValueError("scored measurement differs from the original registered look")
                evidence = detail["identity"]
            if evidence is None:
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("unavailable look requires its disposition")
            else:
                _check(schema("common")["$defs"]["hash"], evidence, {})
                if reason is not None:
                    raise ValueError("measured look cannot also be unavailable")
            previous = self.binding["identity"]
            for earlier in series[:series.index(look)]:
                path = self.root / ("look-" + earlier["identity"][7:] + ".json")
                if not path.is_file():
                    raise ValueError("previous registered look is missing")
                entry = load(path)
                if entry != seal(entry) or entry["look"] != earlier["identity"] or entry["previous"] != previous:
                    raise ValueError("previous registered observation differs")
                previous = entry["identity"]
            path = self.root / ("look-" + look["identity"][7:] + ".json")
            prior = load(path) if path.is_file() else None
            record = dict(version="round029-observation/v1", registration=reg["identity"],
                          previous=previous, look=look["identity"], denominator=len(look["members"]),
                          evidence=evidence, reason=reason)
            if (detail is not None and prior is None) or (prior and "measurement_format" in prior):
                record["measurement_format"] = "round029-measured-look/v1"
            record = seal(record)
            if prior is not None and record != prior:
                raise ValueError("original recovery publication differs: " + str(path))
            if detail is not None:
                measurement = confined(self.root, "measurements/" + evidence[7:] + ".json", exists=self.readonly)
                if self.readonly:
                    if load(measurement, maximum=128 << 20) != detail:
                        raise ValueError("original trial measurement differs")
                else:
                    retain(measurement, detail)
            if self.readonly:
                if prior != record:
                    raise ValueError("original trial observation missing or altered")
            else:
                retain(path, record)
            return record["identity"]

    def velocity_report(self, registration, *, stratum, seed, draws=10_000, budgets=None, model_seed=None):
        """Reopen measured looks and explicit dispositions at one frozen study boundary."""
        from .contracts import CAMPAIGN_DOCUMENT_BYTES, _check, schema
        from .graph import velocity_study_looks
        from .velocity_metrics import measured_look, study_curve
        from oeis_learn.pipeline_validation.identity import confined

        graph = velocity_study_looks(registration)
        model_seed = registration["initialization"] if model_seed is None else model_seed
        selected, measurements = {}, {}
        with self.lock():
            if (load(confined(self.root, "velocity-registration.json")) != registration
                    or self.binding["study"] != registration["identity"]):
                raise ValueError("different frozen study journal")
            for method, profiles in graph.items():
                selected[method] = {}
                for profile, series in profiles.items():
                    looks = [look for look in series if look["seed"] == model_seed]
                    selected[method][profile] = looks
                    previous = self.binding["identity"]
                    for look in looks:
                        ref = look["identity"]
                        path = confined(self.root, "look-" + ref[7:] + ".json", exists=False)
                        if not path.is_file():
                            previous = None
                            continue
                        observation = load(path)
                        if (observation != seal(observation) or observation["version"] != "round029-observation/v1"
                                or observation["registration"] != registration["identity"] or previous is None
                                or observation["previous"] != previous or observation["look"] != ref
                                or observation["denominator"] != len(look["members"])):
                            raise ValueError("registered observation chain differs")
                        previous = observation["identity"]
                        evidence = observation["evidence"]
                        if evidence is None:
                            if not isinstance(observation["reason"], str) or not observation["reason"].strip():
                                raise ValueError("unavailable look requires a disposition")
                            measurements[ref] = {"reason": observation["reason"]}
                            continue
                        _check(schema("common")["$defs"]["hash"], evidence, {})
                        detail = confined(self.root, "measurements/" + evidence[7:] + ".json", exists=False)
                        if not detail.is_file():
                            if "measurement_format" in observation:
                                raise ValueError("committed measurement is missing")
                            measurements[ref] = {"reason": "historical evidence reference without measurement details"}
                            continue
                        value = load(detail, maximum=CAMPAIGN_DOCUMENT_BYTES)
                        expected = measured_look(value["probe"], value["scores"], cut=value["cut"], work=value["work"])
                        if (value != expected or value["identity"] != evidence or value["probe"]["look"] != look
                                or observation["reason"] is not None):
                            raise ValueError("saved scored measurement differs from its registered observation")
                        measurements[ref] = value
        if not any(looks for profiles in selected.values() for looks in profiles.values()):
            raise ValueError("model seed outside the frozen study")
        return study_curve(selected, measurements, stratum=stratum, seed=seed, draws=draws, budgets=budgets)

    def record_velocity_branch(self, branch, *, decision, reason, grant=None, approval=None, root_id=None,
                               accepted=None):
        """Record an unused or independently funded branch without terminal optimizer feedback."""
        from .campaign import velocity_policy
        from .contracts import _check, schema
        from .velocity_records import accepted_grant_parent, validate_grant
        from oeis_learn.pipeline_validation.preparation_recovery import retain

        if branch not in velocity_policy()["branches"] or not isinstance(reason, str) or not reason.strip():
            raise ValueError("registered branch and reason required")
        _check(schema("common")["$defs"]["hash"], decision, {})
        with self.lock():
            reg = load(self.root / "velocity-registration.json")
            if grant is not None:
                if (accepted is None or accepted.get("registration") != reg["identity"]
                        or branch not in grant["allowed_operations"]):
                    raise ValueError("branch missing from independently accepted and funded registration")
                action = "cpu" if grant["scope"] == "cpu" else "run"
                parent = accepted_grant_parent(accepted, grant["scope"])
                validate_grant(grant, approval, root=root_id, parent=parent, action=action)
            record = seal(dict(version="round029-branch/v1", study=reg["identity"], branch=branch,
                               state="funded" if grant else "unused", decision=decision, reason=reason,
                               grant=grant["identity"] if grant else None))
            retain(self.root / ("branch-" + branch + "-" + decision[7:] + ".json"), record)
            return record

    @contextmanager
    def lock(self):
        import fcntl

        p = self.root / "journal.lock"
        if p.is_symlink():
            raise ValueError("lock alias")
        with p.open("rb" if self.readonly else "a") as f:
            fcntl.flock(f, fcntl.LOCK_SH if self.readonly else fcntl.LOCK_EX)
            yield

    def read(self):
        events = sorted(self.root.glob("event-*.json"))
        if len(events) > 4096:
            raise ValueError("trial journal exhausted")
        self.trials = {}
        self.tip = self.binding["identity"]
        for i, p in enumerate(events):
            e = load(p)
            if (
                p.name != f"event-{i:04d}.json"
                or set(e) != {"previous", "ordinal", "state", "identity"}
                or seal(e) != e
                or e["previous"] != self.tip
            ):
                raise ValueError("trial event chain differs")
            record(
                "trial",
                e["state"],
                created_at="2026-11-08T00:00:00Z",
                purpose="acceptance",
                track="synthetic_strict",
            )
            if (
                e["state"]["proposal"]["study"] != self.binding["study"]
                or str(e["state"]["proposal"]["ordinal"]) != e["ordinal"]
            ):
                raise ValueError("trial study or ordinal differs")
            if e["state"]["status"] == "COMPLETE":
                from .trial_evidence import eligible
                if eligible(e["state"]["result"],root=self.provenance_root) is None:
                    raise ValueError("journal contains ineligible complete feedback")
            self.tip = e["identity"]
            self.trials[e["ordinal"]] = e["state"]
        self.count = len(events)

    def write(self, ordinal, state):
        if self.count >= 4096:
            raise ValueError("bounded trial events exhausted")
        record(
            "trial",
            state,
            created_at="2026-11-08T00:00:00Z",
            purpose="acceptance",
            track="synthetic_strict",
        )
        e = seal({"previous": self.tip, "ordinal": str(ordinal), "state": state})
        atomic_write(self.root / f"event-{self.count:04d}.json", e)
        self.read()
        return copy.deepcopy(state)

    def proposal(self, ordinal):
        integer(ordinal)
        self.read()
        return copy.deepcopy(self.trials[str(ordinal)]["proposal"])

    def propose(self, ordinal, *, parameters, seed):
        integer(ordinal)
        integer(seed)
        if ordinal >= 16 or not isinstance(parameters, dict) or not parameters:
            raise ValueError("bounded proposal required")
        digest(parameters)  # reject nonfinite/non-JSON settings
        proposal = seal(
            {
                "study": self.binding["study"],
                "ordinal": ordinal,
                "parameters": parameters,
                "seed": seed,
            }
        )
        with self.lock():
            self.read()
            if str(ordinal) in self.trials:
                if self.trials[str(ordinal)]["proposal"] != proposal:
                    raise ValueError("conflicting duplicate proposal")
                return proposal
            if ordinal != len(self.trials):
                raise ValueError("ordered proposals required")
            self.write(
                ordinal,
                {
                    "proposal": proposal,
                    "status": "PROPOSED",
                    "recovery_pending": False,
                    "tokens": 0,
                    "elapsed_us": 0,
                    "invocation": 0,
                    "result": None,
                },
            )
        return proposal

    def reserve(self, ordinal):
        with self.lock():
            self.read()
            s = copy.deepcopy(self.trials[str(ordinal)])
            if s["status"] != "PROPOSED":
                raise ValueError("proposal already reserved")
            s["status"] = "RESERVED"
            return self.write(ordinal, s)

    def dispatch(self, ordinal):
        with self.lock():
            self.read()
            if str(ordinal) not in self.trials:
                raise ValueError("proposal must be durable before dispatch")
            s = copy.deepcopy(self.trials[str(ordinal)])
            if s["status"] not in {"PROPOSED", "RESERVED"} and not (
                s["status"] == "RUNNING" and s["recovery_pending"]
            ):
                raise ValueError("trial cannot restart or dispatch twice")
            if (
                s["tokens"] >= self.binding["token_cap"]
                or s["elapsed_us"] >= self.binding["elapsed_cap_us"]
            ):
                raise ValueError("trial allowance exhausted")
            s.update(status="RUNNING", recovery_pending=False, invocation=s["invocation"] + 1)
            return self.write(ordinal, s)

    def interrupt(self, ordinal, *, tokens, elapsed_us):
        with self.lock():
            self.read()
            s = copy.deepcopy(self.trials[str(ordinal)])
            self.cost(s, tokens, elapsed_us)
            if s["status"] != "RUNNING" or s["recovery_pending"]:
                raise ValueError("no active invocation")
            s.update(tokens=tokens, elapsed_us=elapsed_us, recovery_pending=True)
            if tokens >= self.binding["token_cap"] or elapsed_us >= self.binding["elapsed_cap_us"]:
                s.update(status="PARTIAL", recovery_pending=False)
            return self.write(ordinal, s)

    def cost(self, s, tokens, elapsed_us):
        integer(tokens)
        integer(elapsed_us)
        if (
            tokens < s["tokens"]
            or elapsed_us < s["elapsed_us"]
            or tokens > self.binding["token_cap"]
            or elapsed_us > self.binding["elapsed_cap_us"]
        ):
            raise ValueError("trial costs must be cumulative within reservation")

    def tell(self, ordinal, result):
        if (
            set(result) - {"evidence"} != {"status", "tokens", "elapsed_us", "value", "role"}
            or result["status"] not in TERMINAL
            or result["role"] != "development"
        ):
            raise ValueError("closed development-only terminal result required")
        if result["value"] is not None:
            scalar(result["value"])
        if result["status"] == "COMPLETE" and result["value"] is None:
            raise ValueError("complete trial needs measured capability")
        from .trial_evidence import eligible

        if result["status"] == "COMPLETE" and eligible(result, root=self.provenance_root) is None:
            raise ValueError("complete trial lacks eligible bound development evidence")
        with self.lock():
            self.read()
            s = copy.deepcopy(self.trials[str(ordinal)])
            if s["result"] is not None:
                if digest(s["result"]) != digest(result):
                    raise ValueError("conflicting duplicate tell")
                return s["result"]
            if s["status"] != "RUNNING" or s["recovery_pending"]:
                raise ValueError("terminal trial or inactive invocation")
            self.cost(s, result["tokens"], result["elapsed_us"])
            s.update(
                status=result["status"],
                tokens=result["tokens"],
                elapsed_us=result["elapsed_us"],
                result=result,
            )
            self.write(ordinal, s)
            return copy.deepcopy(result)

    def equal_token_eligible(self, *, target):
        integer(target, 1)
        self.read()
        from .trial_evidence import eligible

        return [
            int(k)
            for k, s in self.trials.items()
            if s["status"] == "COMPLETE"
            and s["tokens"] >= target
            and eligible(s["result"], root=self.provenance_root) is not None
        ]


def historical_trials(root, folder, study):
    """Read-only snapshot of a closed original study, without creating a binding or a lock."""
    from oeis_learn.pipeline_validation.identity import confined
    binding = load(confined(root, Path(folder) / "binding.json"))
    if (seal(binding) != binding or binding["version"] != "progressive-trial-journal/v1"
            or binding["study"] != study):
        raise ValueError("original study binding differs")
    journal = TrialJournal.__new__(TrialJournal)
    journal.root = confined(root, folder)
    journal.provenance_root = root
    journal.binding = binding
    journal.read()
    return journal.trials, journal.tip


def paired_generator_exposure(ordinary, treatment):
    """A descriptive or funded comparison uses equal useful and charged trajectories."""
    from .training_rows import GENERATOR_LOOKS, pair_seeds
    from .contracts import seal
    import re
    both = ordinary, treatment
    fields = {"version", "identity", "parent", "registration", "pair", "seeds", "arm", "stream", "initial",
              "codec", "common_policy", "target", "charged", "discarded", "overshoot", "order_digest", "useful_tokens",
              "last_update_tokens", "updates", "cursor"}
    for arm, cut in zip(("ordinary", "first_divergence"), both):
        if (set(cut) != fields or cut["version"] != "round028-generator-milestone/v1" or seal(cut) != cut
                or cut["arm"] != arm or cut["seeds"] != pair_seeds(cut["registration"], cut["pair"])
                or type(cut["target"]) is not int or cut["target"] not in GENERATOR_LOOKS):
            raise ValueError("registered arm and original matched milestone required")
        if any(not isinstance(cut[k], str) or not re.fullmatch("sha256:[0-9a-f]{64}", cut[k])
               for k in ("parent", "stream", "initial", "codec", "common_policy", "order_digest")):
            raise ValueError("content-bound training exposure required")
        if (any(type(cut[k]) is not int or cut[k] < 0 for k in ("charged", "discarded", "overshoot", "useful_tokens",
                                                                "last_update_tokens", "updates", "cursor"))
                or cut["useful_tokens"] + cut["discarded"] != cut["charged"]
                or cut["cursor"] != 32 * cut["updates"] or cut["overshoot"] != max(0, cut["charged"]-cut["target"])):
            raise ValueError("inconsistent matched milestone accounting")
    if any(ordinary[k] != treatment[k] for k in fields - {"identity", "parent", "arm"}):
        return "UNEQUAL_EXPOSURE"
    if ordinary["charged"] < ordinary["target"]:
        return "INCOMPLETE_LOOK"
    if ordinary["target"] == 0:
        return "MATCHED" if ordinary["charged"] == ordinary["updates"] == 0 else "MISSED_LOOK"
    if not 0 <= ordinary["overshoot"] < ordinary["last_update_tokens"]:
        return "MISSED_LOOK"
    return "MATCHED"

def reusable_zero_look(ordinary, treatment, requests, executions):
    """Credit one prior zero-look execution only on equality of all inference inputs."""
    from .contracts import digest
    import re
    if paired_generator_exposure(ordinary, treatment) != "MATCHED" or ordinary["target"] != 0:
        return False
    fields = {"weights", "ordered_prompts", "codec", "inference_seed", "runtime", "source", "profile",
              "fit_and_choice", "budgets", "support", "comparator", "role"}
    comparisons = []
    for arm in ("ordinary", "first_divergence"):
        request, evidence = requests[arm], executions[arm]
        if (request["target"] != 0 or request["arm"] != arm or request["registration"] != ordinary["registration"]
                or request["seed"] != ordinary["seeds"]["initialization"] or set(evidence) != fields
                or evidence["weights"] != ordinary["initial"] or evidence["codec"] != ordinary["codec"]
                or type(evidence["inference_seed"]) is not int or evidence["inference_seed"] < 0
                or any(not isinstance(v, str) or not re.fullmatch("sha256:[0-9a-f]{64}", v)
                       for k, v in evidence.items() if k != "inference_seed")):
            return False
        comparisons.append(digest([{k:v for k,v in request.items() if k not in ("arm","identity")}, evidence]))
    return comparisons[0] == comparisons[1]
