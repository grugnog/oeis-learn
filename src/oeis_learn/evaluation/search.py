"""Visible-only bounded search. Every expansion and attempt consumes one ledger.

The service accepts a proposer, not evaluator truth. Distinct sources are never
pruned by their outputs on a finite prefix. Beam paths use the learned policy;
repair is a separate, explicitly labelled source transformation.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import time

import torch

from oeis_learn.decoder.program_codec import BOS_ID, EOS_ID, decode_body, encode_body
from oeis_learn.decoder.environment_tracker import FoundationEnvironmentTracker
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.rl.policy import masked_distribution, scoring_mode


class SearchExhausted(RuntimeError):
    pass


@dataclass
class SearchLedger:
    max_attempts: int = 32
    max_expansions: int = 4096
    max_tokens: int = 32768
    budget_ns: int = 60_000_000_000
    attempts: int = 0
    expansions: int = 0
    tokens: int = 0
    charged_ns: int = 0
    _started: int = field(default_factory=time.monotonic_ns, repr=False)

    def __post_init__(self):
        for name, cap in (
            ("max_attempts", 32),
            ("max_expansions", 100000),
            ("max_tokens", 32768),
            ("budget_ns", 300_000_000_000),
        ):
            if type(getattr(self, name)) is not int or not 0 < getattr(self, name) <= cap:
                raise ValueError("bounded exact search limits required")
        if any(
            type(getattr(self, n)) is not int or getattr(self, n) < 0
            for n in ("attempts", "expansions", "tokens", "charged_ns")
        ):
            raise ValueError("invalid charged search state")

    def elapsed(self):
        return self.charged_ns + time.monotonic_ns() - self._started

    def charge(self, *, attempts=0, expansions=0, tokens=0):
        if any(type(v) is not int or v < 0 for v in (attempts, expansions, tokens)):
            raise ValueError("nonnegative charges required")
        if (
            self.elapsed() >= self.budget_ns
            or self.attempts + attempts > self.max_attempts
            or self.expansions + expansions > self.max_expansions
            or self.tokens + tokens > self.max_tokens
        ):
            raise SearchExhausted("shared search budget exhausted")
        self.attempts += attempts
        self.expansions += expansions
        self.tokens += tokens

    def state_dict(self):
        return {
            "version": "search-ledger/v1",
            **{
                k: getattr(self, k)
                for k in (
                    "max_attempts",
                    "max_expansions",
                    "max_tokens",
                    "budget_ns",
                    "attempts",
                    "expansions",
                    "tokens",
                )
            },
            "charged_ns": self.elapsed(),
        }

    @classmethod
    def from_state(cls, state):
        if (
            set(state)
            != {
                "version",
                "max_attempts",
                "max_expansions",
                "max_tokens",
                "budget_ns",
                "attempts",
                "expansions",
                "tokens",
                "charged_ns",
            }
            or state["version"] != "search-ledger/v1"
        ):
            raise ValueError("incompatible search ledger")
        result = cls(**{k: v for k, v in state.items() if k != "version"})
        if (
            result.attempts > result.max_attempts
            or result.expansions > result.max_expansions
            or result.tokens > result.max_tokens
        ):
            raise ValueError("search charges exceed limits")
        return result


def typed_repairs(source, limit=8):
    """Local literal neighbours and same-stack-effect operator changes only."""
    from oeis_learn.sandbox.wat_ast import parse

    parse(source)
    words, result = source.split(), []
    swaps = {
        "i32.add": "i32.sub",
        "i32.sub": "i32.add",
        "i64.add": "i64.sub",
        "i64.sub": "i64.add",
        "i256.add": "i256.sub",
        "i256.sub": "i256.add",
    }
    for i, word in enumerate(words):
        replacements = []
        if word.endswith(".const") and i + 1 < len(words):
            replacements = [(i + 1, str(int(words[i + 1]) + delta)) for delta in (-1, 1)]
        elif word in swaps:
            replacements = [(i, swaps[word])]
        for at, value in replacements:
            variant = words[:]
            variant[at] = value
            try:
                candidate = decode_body(encode_body(" ".join(variant)))
                parse(candidate)
            except ValueError:
                continue
            if candidate != source and candidate not in result:
                result.append(candidate)
                if len(result) == limit:
                    return tuple(result)
    return tuple(result)


class ModelProposer:
    def __init__(self, encoder, decoder, *, context_limit=1024):
        self.encoder, self.decoder, self.context_limit = encoder, decoder, context_limit

    def stochastic(self, visible, seed, ledger, *, greedy=False):
        from oeis_learn.decoder.sampler import WatProgramSampler

        device = next(self.encoder.parameters()).device
        sampler = WatProgramSampler(
            self.decoder,
            max_length=self.context_limit - 1,
            temperature=0 if greedy else 0.8,
            top_p=1,
            codec_profile="wat_body_decimal_v1",
            use_cache=True,
        )
        with scoring_mode(self.encoder, self.decoder), torch.no_grad():
            memory = self.encoder.forward_from_sequences([visible], device=device)
            trajectory = sampler.sample_trajectories(
                memory, [seed], budget_hook=lambda n: ledger.charge(expansions=n, tokens=n)
            )[0]
        return trajectory.source, {
            "method": "greedy" if greedy else "stochastic",
            "tokens": list(trajectory.tokens),
            "complete": trajectory.complete,
            "failure": trajectory.failure,
            "trajectory": {
                "tokens": list(trajectory.tokens),
                "supports": [list(x) for x in trajectory.supports],
                "log_probabilities": list(trajectory.log_probabilities),
                "forced": list(trajectory.forced),
                "temperature": trajectory.temperature,
                "seed": trajectory.seed,
            },
            "encoder_sha256": __import__(
                "oeis_learn.rl.policy", fromlist=["model_digest"]
            ).model_digest(self.encoder),
            "decoder_sha256": __import__(
                "oeis_learn.rl.policy", fromlist=["model_digest"]
            ).model_digest(self.decoder),
        }

    def beam(self, visible, ledger, count=8, width=8):
        device = next(self.encoder.parameters()).device
        beams, complete = [(0.0, [BOS_ID], FoundationEnvironmentTracker())], []
        with scoring_mode(self.encoder, self.decoder), torch.no_grad():
            memory = self.encoder.forward_from_sequences([visible], device=device)
            for _ in range(self.context_limit - 1):
                candidates = []
                for score, tokens, tracker in beams:
                    legal = tracker.get_valid_next_tokens()
                    if not legal:
                        continue
                    ledger.charge(expansions=1)
                    logits = self.decoder(torch.tensor([tokens], device=device), memory)[0, -1]
                    support = torch.zeros_like(logits, dtype=torch.bool)
                    support[legal] = True
                    logs, _, _ = masked_distribution(logits, support, 0.8)
                    ordered = sorted(legal, key=lambda t: (-float(logs[t]), t))[:width]
                    ledger.charge(tokens=len(ordered))
                    for token in ordered:
                        new_tokens, new_tracker = tokens + [token], deepcopy(tracker)
                        from oeis_learn.decoder.program_codec import ID_TO_TOKEN

                        new_tracker.update(ID_TO_TOKEN[token])
                        entry = (score + float(logs[token]), new_tokens, new_tracker)
                        if token == EOS_ID:
                            complete.append(entry)
                        else:
                            candidates.append(entry)
                beams = sorted(candidates, key=lambda row: (-row[0], row[1]))[:width]
                if len(complete) >= count or not beams:
                    break
        return tuple(
            (
                decode_body(tokens[1:]),
                {
                    "method": "beam",
                    "tokens": tokens,
                    "score": score,
                    "complete": True,
                    "encoder_sha256": __import__(
                        "oeis_learn.rl.policy", fromlist=["model_digest"]
                    ).model_digest(self.encoder),
                    "decoder_sha256": __import__(
                        "oeis_learn.rl.policy", fromlist=["model_digest"]
                    ).model_digest(self.decoder),
                },
            )
            for score, tokens, _ in sorted(complete, key=lambda row: (-row[0], row[1]))[:count]
        )


class SearchSession:
    def __init__(
        self, visible, *, ledger=None, portfolio="portfolio32", seed=20260913, retrieval=()
    ):
        from oeis_learn.experiments.profiles import I256_MIN, I256_MAX

        if len(visible) != 20 or any(
            type(v) is not int or not I256_MIN <= v <= I256_MAX for v in visible
        ):
            raise ValueError("search sees exactly twenty signed256 values")
        if portfolio not in ("portfolio32", "baseline16", "greedy"):
            raise ValueError("frozen supported portfolio required")
        self.visible = tuple(visible)
        self.ledger = ledger or SearchLedger(
            max_attempts=16 if portfolio == "baseline16" else 1 if portfolio == "greedy" else 32
        )
        maximum = 32 if portfolio == "portfolio32" else 16 if portfolio == "baseline16" else 1
        if self.ledger.max_attempts != maximum:
            raise ValueError("portfolio attempt allocation and shared budget differ")
        self.portfolio, self.seed = portfolio, seed
        self.cursor, self.results, self.seen, self.frontier = 0, [], set(), []
        if retrieval:
            raise ValueError("retrieval must use a verified frozen archive capability")
        self.retrieval = ()
        self.archive_snapshot = None
        self.retrieval_keys = {}

    @classmethod
    def with_archive(cls, visible, archive, snapshot, **kwargs):
        # Retrieval can only come from the expanded, verified frozen training view.
        obj = cls(visible, **kwargs)
        obj.retrieval = tuple(value["source"] for _, value in archive.training_view(snapshot))
        obj.retrieval_keys = {
            value["source"]: key for key, value in archive.training_view(snapshot)
        }
        obj.archive_snapshot = deepcopy(snapshot)
        obj._archive = archive
        return obj

    def _execute(self, source, provenance, runner):
        canonical = decode_body(encode_body(source))
        key = digest(canonical)
        if key in self.seen:
            return {"status": "duplicate_source", "source": canonical, "provenance": provenance}
        self.seen.add(key)
        self.ledger.charge()  # deadline before every final execution
        evidence = runner.run_single(
            canonical,
            list(range(20)),
            request_id=f"search-{self.seed}-{self.cursor}",
            deadline_ns=time.monotonic_ns() + max(0, self.ledger.budget_ns - self.ledger.elapsed()),
        ).to_dict()
        if (
            provenance.get("method") == "stochastic"
            and provenance.get("complete")
            and "trajectory" in provenance
        ):
            from oeis_learn.decoder.sampler import SampledTrajectory

            values = provenance["trajectory"]
            trajectory = SampledTrajectory(
                canonical,
                tuple(values["tokens"]),
                tuple(tuple(s) for s in values["supports"]),
                tuple(values["log_probabilities"]),
                tuple(values["forced"]),
                values["temperature"],
                values["seed"],
                True,
                None,
            )
            rollout = trajectory.record(
                encoder_sha256=provenance["encoder_sha256"],
                decoder_sha256=provenance["decoder_sha256"],
                prompt_sha256=digest([str(v) for v in self.visible]),
                execution_sha256=digest(evidence),
                reward_components={
                    "fit": float(evidence["outputs"] == [str(v) for v in self.visible]),
                    "validity": float(evidence["outcome"] is None),
                    "cost": float(evidence.get("fuel") or 0),
                },
                eos_id=EOS_ID,
            )
            provenance = dict(provenance, rollout=rollout)
        if "encoder_sha256" in provenance and "decoder_sha256" in provenance:
            from oeis_learn.sandbox.pipeline import digest_bytes

            hypothesis = {
                "version": "search-hypothesis/v1",
                "method": provenance["method"],
                "source_sha256": digest_bytes(canonical.encode()),
                "body_tokens": encode_body(canonical),
                "visible_prompt_sha256": digest([str(v) for v in self.visible]),
                "encoder_sha256": provenance["encoder_sha256"],
                "decoder_sha256": provenance["decoder_sha256"],
                "execution": evidence,
                "parent": provenance.get("parent"),
            }
            hypothesis["identity_sha256"] = digest(hypothesis)
            provenance = dict(provenance, hypothesis=hypothesis)
        matched = (
            evidence["outcome"] is None
            and evidence["outputs"] == [str(v) for v in self.visible]
            and evidence["reference_outputs"] == evidence["outputs"]
        )
        return {
            "status": "visible_fit" if matched else "visible_mismatch",
            "source": canonical,
            "execution": evidence,
            "provenance": provenance,
            "counterexamples": [
                i for i, (a, b) in enumerate(zip(evidence["outputs"], self.visible)) if a != str(b)
            ],
        }

    def step(self, proposer, runner):
        maximum = (
            32 if self.portfolio == "portfolio32" else 16 if self.portfolio == "baseline16" else 1
        )
        if self.cursor >= maximum:
            return None
        self.ledger.charge(attempts=1)
        slot = self.cursor
        self.cursor += 1  # spent attempts are never refunded
        try:
            if self.portfolio == "greedy" or slot < 16:
                source, provenance = proposer.stochastic(
                    self.visible, self.seed + slot, self.ledger, greedy=self.portfolio == "greedy"
                )
            elif slot < 24:
                if slot == 16:
                    self.frontier = list(proposer.beam(self.visible, self.ledger, count=8))
                source, provenance = (
                    self.frontier.pop(0)
                    if self.frontier
                    else ("", {"method": "beam", "failure": "no_complete_path"})
                )
            else:
                parents = [
                    r["source"]
                    for r in self.results
                    if r.get("source")
                    and r.get("status") != "duplicate_source"
                    and r.get("provenance", {}).get("method") != "typed_repair"
                ] + list(self.retrieval)
                variants = [
                    (source, variant) for source in parents for variant in typed_repairs(source)
                ]
                offset = slot - 24
                if offset >= len(variants):
                    source, provenance = (
                        "",
                        {"method": "typed_repair", "failure": "no_typed_neighbour"},
                    )
                else:
                    parent, source = variants[offset]
                    self.ledger.charge(expansions=1, tokens=len(encode_body(source)))
                    parent_result = next(
                        (
                            r
                            for r in self.results
                            if r.get("source") == parent and "hypothesis" in r.get("provenance", {})
                        ),
                        None,
                    )
                    identities = next(
                        (
                            r["provenance"]
                            for r in self.results
                            if "encoder_sha256" in r.get("provenance", {})
                        ),
                        None,
                    )
                    provenance = {"method": "typed_repair", "parent_source": parent}
                    if identities is not None and (
                        parent_result is not None or parent in self.retrieval_keys
                    ):
                        provenance.update(
                            encoder_sha256=identities["encoder_sha256"],
                            decoder_sha256=identities["decoder_sha256"],
                        )
                        provenance["parent"] = (
                            {
                                "kind": "model",
                                "hypothesis": parent_result["provenance"]["hypothesis"],
                            }
                            if parent_result
                            else {
                                "kind": "archive",
                                "key": self.retrieval_keys[parent],
                                "snapshot": deepcopy(self.archive_snapshot),
                            }
                        )
            result = (
                self._execute(source, provenance, runner)
                if source
                else {"status": "incomplete", "provenance": provenance}
            )
        except SearchExhausted:
            result = {"status": "budget_exhausted"}
        self.results.append(result)
        return deepcopy(result)

    def run(self, proposer, runner):
        while self.cursor < self.ledger.max_attempts:
            try:
                self.step(proposer, runner)
            except SearchExhausted:
                break
        return self.report()

    def report(self):
        return {
            "version": "bounded-search/v1",
            "label": "model-only" if self.portfolio in ("baseline16", "greedy") else "model+search",
            "portfolio": self.portfolio,
            "requested_attempts": 32
            if self.portfolio == "portfolio32"
            else 16
            if self.portfolio == "baseline16"
            else 1,
            "spent": self.ledger.state_dict(),
            "results": deepcopy(self.results),
            "comparison_key": digest(
                [
                    self.ledger.max_attempts,
                    self.ledger.max_expansions,
                    self.ledger.max_tokens,
                    self.ledger.budget_ns,
                ]
            ),
        }

    def state_dict(self):
        return {
            "version": "search-session/v1",
            "visible": list(self.visible),
            "portfolio": self.portfolio,
            "seed": self.seed,
            "cursor": self.cursor,
            "results": deepcopy(self.results),
            "seen": sorted(self.seen),
            "frontier": deepcopy(self.frontier),
            "retrieval": list(self.retrieval),
            "archive_snapshot": deepcopy(self.archive_snapshot),
            "retrieval_keys": dict(self.retrieval_keys),
            "ledger": self.ledger.state_dict(),
        }

    @classmethod
    def from_state(cls, state, *, archive=None):
        if state["version"] != "search-session/v1":
            raise ValueError("incompatible search session")
        obj = cls(
            state["visible"],
            ledger=SearchLedger.from_state(state["ledger"]),
            portfolio=state["portfolio"],
            seed=state["seed"],
        )
        if state["archive_snapshot"] is not None:
            if archive is None:
                raise ValueError("retrieval resume requires the verified archive")
            obj.retrieval = tuple(
                value["source"] for _, value in archive.training_view(state["archive_snapshot"])
            )
            obj.archive_snapshot = deepcopy(state["archive_snapshot"])
            obj.retrieval_keys = {
                value["source"]: key
                for key, value in archive.training_view(state["archive_snapshot"])
            }
        if (
            list(obj.retrieval) != state["retrieval"]
            or obj.retrieval_keys != state["retrieval_keys"]
        ):
            raise ValueError("retrieval resume view differs")
        if (
            type(state["cursor"]) is not int
            or state["cursor"] != state["ledger"]["attempts"]
            or len(state["results"]) != state["cursor"]
        ):
            raise ValueError("incomplete search boundary cannot resume")
        obj.cursor, obj.results, obj.seen, obj.frontier = (
            state["cursor"],
            deepcopy(state["results"]),
            set(state["seen"]),
            deepcopy(state["frontier"]),
        )
        return obj

    def load_state_dict(self, state):
        restored = self.from_state(state, archive=getattr(self, "_archive", None))
        if (
            restored.visible != self.visible
            or restored.portfolio != self.portfolio
            or restored.seed != self.seed
        ):
            raise ValueError("search task/profile differs on resume")
        self.__dict__.update(restored.__dict__)


def value_guided_repairs(
    source,
    value_model,
    *,
    archive,
    training_snapshot,
    measurement_decision,
    ledger,
    max_expansions=32,
):
    """Training-only learned ranking of bounded typed repairs, after a pilot gate.

    Scores rank hypotheses; all changed programs still require normal exact
    execution/admission. This function makes no rewrite-equivalence claim.
    """
    if not 1 <= max_expansions <= 32:
        raise ValueError("value proposer cap32")
    if (
        set(measurement_decision)
        != {"branch", "disposition", "training_snapshot_sha256", "baseline_failure_observed"}
        or measurement_decision["branch"] != "value_guided_repair"
        or measurement_decision["disposition"] != "pilot_selected"
        or measurement_decision["baseline_failure_observed"] is not True
        or measurement_decision["training_snapshot_sha256"] != training_snapshot["root_sha256"]
    ):
        raise ValueError("measured training-only value pilot gate required")
    if len(training_snapshot["entries"]) > 4096:
        raise ValueError("value targets archive snapshot cap4096")
    training_view = dict(archive.training_view(training_snapshot))
    if not training_view or any(
        not e["entry"]["training_eligible"] for e in training_view.values()
    ):
        raise ValueError("verified eligible training targets required")
    targets = [
        {"source": e["source"], "conditioning_values": e["entry"]["conditioning_values"]}
        for e in list(training_view.values())[:64]
    ]
    variants = typed_repairs(source, limit=max_expansions)
    scored = []
    for candidate in variants:
        ledger.charge(expansions=1, tokens=len(encode_body(candidate)))
        score = float(value_model(candidate, deepcopy(targets)))
        if not __import__("math").isfinite(score):
            raise ValueError("finite learned value required")
        scored.append((score, candidate))
    return tuple(source for _, source in sorted(scored, key=lambda item: (-item[0], item[1])))
