"""Fixed visible-only baselines and shared portfolio accounting.

No observed outcome union is an executable policy. Model loading delegates to
existing constructors/loaders; conventional methods disclose their narrow priors.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction

from .cohorts import DEFAULT_PROFILE, validate_profile, validate_prompt
from .contracts import digest, verify_refs

METHODS = (
    "retained_D",
    "random",
    "earlier_checkpoint",
    "shuffled",
    "absent",
    "templates",
    "polynomial",
    "linear_recurrence",
)


@dataclass(frozen=True)
class ConventionalPolicy:
    """Prospective priors; integer coefficients only, recurrence indices 0..19.

    Polynomial indices may have any valid offset. No rational-coefficient or
    affine-recurrence fallback is inferred from outcomes. The existing codec
    imposes its independent 1,024-token ceiling on every returned candidate.
    """

    max_degree: int = 3
    max_order: int = 2
    coefficient_bits: int = 4096

    def __post_init__(self):
        for value, low, high in (
            (self.max_degree, 0, 3),
            (self.max_order, 1, 2),
            (self.coefficient_bits, 1, 4096),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid frozen conventional bound")


def load_baseline(
    method, recipe, *, root=None, ref=None, seed=None, reference_recipe=None, device="cpu"
):
    """Load an actual model control; intervention inputs are handled separately.

    D is just a supplied reference. Earlier weights require a resolved reference
    recipe: matching tensor shapes alone cannot establish input compatibility.
    This does not restore optimizer state or authorize an evaluation campaign.
    """
    if recipe.get("version") == "progressive-research-model/v1":
        from oeis_learn.generalization.progressive.models import construct, validate_profile
        import torch
        from oeis_learn.pipeline_validation.identity import confined

        if method not in {"random", "retained_D", "earlier_checkpoint", "shuffled", "absent"}:
            raise ValueError("registered research model baseline required")
        if method == "earlier_checkpoint" and reference_recipe != recipe:
            raise ValueError("compatible original research recipe required")
        profile = validate_profile(recipe)
        if method == "random":
            if ref is not None or root is not None or seed != profile["model"]["seed"]:
                raise ValueError("bound random research model required")
        elif ref is None or root is None or seed is not None:
            raise ValueError("identified research checkpoint required")
        encoder, decoder, counts = construct(profile, device=device)
        if ref is not None:
            verify_refs(root, [ref], max_bytes=64 << 30)
            state = torch.load(confined(root, ref["path"]), map_location=device, weights_only=True)
            if set(state) != {"encoder", "decoder"}:
                raise ValueError("generator requires the verified weight slice")
            for name, model in (("encoder", encoder), ("decoder", decoder)):
                expected = model.state_dict()
                if set(state[name]) != set(expected) or any(
                    not isinstance(v, torch.Tensor)
                    or v.dtype != expected[k].dtype
                    or not bool(torch.isfinite(v).all())
                    for k, v in state[name].items()
                ):
                    raise ValueError("finite compatible checkpoint tensors required")
                model.load_state_dict(state[name])
        encoder.eval()
        decoder.eval()
        return encoder, decoder, counts
    from oeis_learn.fresh.bigint_config import resolve

    if resolve(recipe) != recipe:
        raise ValueError("fully resolved baseline recipe required")
    if method == "random":
        if ref is not None or root is not None:
            raise ValueError("random baseline cannot load checkpoint weights")
        return load_random(recipe, seed=seed, device=device)
    if method not in {"retained_D", "earlier_checkpoint", "shuffled", "absent"}:
        raise ValueError("registered model baseline required")
    if ref is None or root is None or seed is not None:
        raise ValueError("identified checkpoint and its recipe required")
    if method == "earlier_checkpoint" and reference_recipe is None:
        raise ValueError("earlier checkpoint requires a reference recipe")
    if reference_recipe is not None:
        if resolve(reference_recipe) != reference_recipe:
            raise ValueError("fully resolved reference recipe required")

        def compatibility(value):
            return {
                "version": value["version"],
                "inputs": value["inputs"],
                "execution": value["execution"],
                "model": {k: v for k, v in value["model"].items() if k != "seed"},
                "model_profile": value.get("model_profile"),
            }

        if compatibility(recipe) != compatibility(reference_recipe):
            raise ValueError("incompatible checkpoint input/model/language recipe")
    return load_checkpoint(root, ref, recipe, device=device)


def checkpoint_identity(root, ref):
    verify_refs(root, [ref], max_bytes=64 << 30)
    return ref["sha256"]


def load_checkpoint(root, ref, recipe, *, device="cpu"):
    checkpoint_identity(root, ref)
    from oeis_learn.fresh.bigint_config import resolve

    if resolve(recipe) != recipe:
        raise ValueError("fully resolved checkpoint recipe required")
    from oeis_learn.pipeline_validation.generalization_evaluation import load_retained

    return load_retained(root, ref, recipe, device=device)


def load_random(recipe, *, seed, device="cpu"):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("frozen random seed required")
    from copy import deepcopy

    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_worker import models

    if resolve(recipe) != recipe:
        raise ValueError("fully resolved random-baseline recipe required")
    recipe = deepcopy(recipe)
    recipe.pop("identity")
    recipe["model"]["seed"] = seed
    recipe = resolve(recipe)
    encoder, decoder, counts = models(recipe, device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, counts


def conditioning(prompts, mode, *, seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("frozen conditioning seed required")
    for p in prompts:
        validate_prompt(p)
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("unique visible prompt identities required")
    if mode == "absent":
        return [
            {
                "id": p["id"],
                "indices": list(p["indices"]),
                "values": None,
                "conditioning": "absent_marker",
            }
            for p in prompts
        ]
    if mode == "shuffled":
        if len(prompts) < 2:
            raise ValueError("shuffle needs at least two visible prompts")
        order = list(range(len(prompts)))
        random.Random(seed).shuffle(order)
        # One cycle is a derangement, independently of original roster order.
        donors = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        return [
            {
                "id": p["id"],
                "indices": list(p["indices"]),
                "values": list(prompts[donors[i]]["values"]),
            }
            for i, p in enumerate(prompts)
        ]
    if mode == "normal":
        return [dict(p, indices=list(p["indices"]), values=list(p["values"])) for p in prompts]
    raise ValueError("unregistered conditioning")


@dataclass
class Portfolio:
    """One shared attempt/deadline allowance, including fitting and branch costs."""

    profile: dict = field(default_factory=lambda: dict(DEFAULT_PROFILE))
    started: float = field(default_factory=time.monotonic)
    attempts: int = 0
    costs: list = field(default_factory=list)
    conventional: ConventionalPolicy = field(default_factory=ConventionalPolicy)
    _reservations: list = field(default_factory=list, init=False, repr=False)
    _conventional_identity: str = field(init=False, repr=False)
    _conventional_record: dict = field(init=False, repr=False)

    def __post_init__(self):
        validate_profile(self.profile)
        self.profile = dict(self.profile)
        if not isinstance(self.conventional, ConventionalPolicy):
            raise TypeError("frozen conventional policy required")
        self._conventional_record = asdict(self.conventional)
        self._conventional_identity = digest(self._conventional_record)

    @property
    def deadline(self):
        return self.started + self.profile["portfolio_us"] / 1_000_000

    def reserve(self, method, *, now=None):
        validate_profile(self.profile)
        if digest(asdict(self.conventional)) != self._conventional_identity:
            raise ValueError("frozen conventional policy changed")
        if method not in METHODS:
            raise ValueError("unregistered baseline")
        now = time.monotonic() if now is None else now
        if not math.isfinite(now) or not math.isfinite(self.started) or now < self.started:
            raise ValueError("invalid measured portfolio clock")
        if now >= self.deadline or self.attempts >= self.profile["attempts"]:
            raise TimeoutError("total portfolio allowance exhausted")
        self.attempts += 1
        self._reservations.append(method)
        return min(self.deadline, now + self.profile["fit_us"] / 1_000_000)

    def observe(self, method, *, candidate, started, ended, outcome, phase="generation"):
        if method not in self._reservations:
            raise ValueError("unreserved method cost")
        if phase not in {"generation", "fit"}:
            raise ValueError("registered cost phase required")
        if not math.isfinite(started) or not math.isfinite(ended) or ended < started:
            raise ValueError("invalid measured interval")
        self.costs.append(
            {
                "method": method,
                "phase": phase,
                "candidate": candidate,
                "elapsed_us": int((ended - started) * 1_000_000),
                "outcome": outcome,
            }
        )

    def report(self):
        methods = {
            method: {"attempts": self._reservations.count(method), "candidates": 0, "elapsed_us": 0}
            for method in self._reservations
        }
        for cost in self.costs:
            row = methods[cost["method"]]
            row["candidates"] += int(
                cost["phase"] == "generation" and cost["candidate"] is not None
            )
            row["elapsed_us"] += cost["elapsed_us"]
        return {
            "attempts": self.attempts,
            "conventional_policy": dict(self._conventional_record),
            "conventional_identity": self._conventional_identity,
            "methods": methods,
            "details": list(self.costs),
            "total_elapsed_us": max(0, int((time.monotonic() - self.started) * 1_000_000)),
            "accounting": "one shared portfolio; overlapping method timings are not added to wall time",
        }


def service_outcome(result, *, observed_at, deadline, fitting_supported=True):
    if observed_at >= deadline:
        return {"success": False, "complete": False, "reason": "expired_service_result"}
    if not fitting_supported:
        return {"success": False, "complete": False, "reason": "unsupported_fitting"}
    return result


def _solve(matrix, values):
    n = len(values)
    a = [[Fraction(x) for x in row] + [Fraction(y)] for row, y in zip(matrix, values)]
    if any(len(row) != n + 1 for row in a):
        return None
    for col in range(n):
        pivot = next((i for i in range(col, n) if a[i][col]), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for i in range(n):
            if i == col:
                continue
            scale = a[i][col]
            a[i] = [x - scale * y for x, y in zip(a[i], a[col])]
    return [row[-1] for row in a]


def polynomial(prompt, *, max_degree=3, coefficient_bits=4096):
    validate_prompt(prompt)
    ConventionalPolicy(max_degree=max_degree, coefficient_bits=coefficient_bits)
    xs = [int(x) for x in prompt["indices"]]
    ys = [int(x) for x in prompt["values"]]
    for degree in range(max_degree + 1):
        coefficients = _solve(
            [[x**j for j in range(degree + 1)] for x in xs[: degree + 1]], ys[: degree + 1]
        )
        if coefficients is None or any(
            c.denominator != 1 or abs(c.numerator).bit_length() > coefficient_bits
            for c in coefficients
        ):
            continue
        if any(sum(c * x**j for j, c in enumerate(coefficients)) != y for x, y in zip(xs, ys)):
            continue
        source = f"(int {int(coefficients[-1])})"
        for coefficient in reversed(coefficients[:-1]):
            source = f"(add (mul {source} $index) (int {int(coefficient)}))"
        return "(program (return " + source + "))"
    return None


def linear_recurrence(prompt, *, max_order=2, coefficient_bits=4096):
    validate_prompt(prompt)
    ConventionalPolicy(max_order=max_order, coefficient_bits=coefficient_bits)
    if prompt["indices"] != [str(i) for i in range(20)]:
        return None
    y = [int(v) for v in prompt["values"]]
    # The all-zero sequence has a valid homogeneous recurrence, but its
    # coefficient system is singular. Choose the fixed zero-coefficient case.
    if not any(y):
        return "(program (return (int 0)))"
    for order in range(1, max_order + 1):
        coeff = _solve(
            [[y[n - j - 1] for j in range(order)] for n in range(order, 2 * order)],
            y[order : 2 * order],
        )
        if coeff is None or any(
            c.denominator != 1 or abs(c.numerator).bit_length() > coefficient_bits for c in coeff
        ):
            continue
        if any(
            sum(coeff[j] * y[n - j - 1] for j in range(order)) != y[n] for n in range(order, 20)
        ):
            continue
        if order == 1:
            return f"(program (local $b0 bigint) (local $c0 control) (set $b0 (int {y[0]})) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (int {int(coeff[0])})))) (return (get $b0)))"
        return f"(program (local $b0 bigint) (local $b1 bigint) (local $b2 bigint) (local $c0 control) (set $b0 (int {y[0]})) (set $b1 (int {y[1]})) (loop $c0 (to_control $index) (set $b2 (add (mul (int {int(coeff[0])}) (get $b1)) (mul (int {int(coeff[1])}) (get $b0)))) (set $b0 (get $b1)) (set $b1 (get $b2))) (return (get $b0)))"
    return None


def iter_candidates(method, prompt, portfolio):
    """Drainable visible-only sources under one prospectively fixed allowance.

    The caller fits/drains each yielded candidate before requesting another.
    Generation, errors and inapplicability all consume an attempt. Fit service
    starts after generation, bounded by the same overall portfolio deadline.
    Model generation and final native/scorer execution belong to T027.
    """
    from oeis_learn.decoder.bigint_codec import CodecError, encode_body

    validate_prompt(prompt)
    if method == "templates":
        from oeis_learn.fresh.template_baseline import template_bank

        sources = template_bank()["sources"]
    elif method in {"polynomial", "linear_recurrence"}:
        sources = [None]
    else:
        raise ValueError("model candidates require the installed generator")
    for source in sources:
        began = time.monotonic()
        try:
            portfolio.reserve(method)
        except TimeoutError:
            break
        reason = None
        try:
            policy = portfolio.conventional
            if method == "polynomial":
                source = polynomial(
                    prompt, max_degree=policy.max_degree, coefficient_bits=policy.coefficient_bits
                )
            elif method == "linear_recurrence":
                source = linear_recurrence(
                    prompt, max_order=policy.max_order, coefficient_bits=policy.coefficient_bits
                )
            if source is None:
                reason = "outside_frozen_applicability"
            else:
                try:
                    encode_body(source)
                except CodecError:
                    source, reason = None, "codec_limit"
        except Exception:
            portfolio.observe(
                method, candidate=None, started=began, ended=time.monotonic(), outcome="error"
            )
            raise
        ended = time.monotonic()
        expired = ended >= portfolio.deadline
        portfolio.observe(
            method,
            candidate=digest(source) if source is not None else None,
            started=began,
            ended=ended,
            outcome="deadline" if expired else "candidate_generated" if source else "inapplicable",
        )
        if expired:
            break
        yield {
            "method": method,
            "source": source,
            "fit_deadline": min(
                portfolio.deadline, ended + portfolio.profile["fit_us"] / 1_000_000
            ),
            "applicable": source is not None,
            "reason": reason,
            "conventional_identity": portfolio._conventional_identity,
        }


def candidates(method, prompt, portfolio):
    """Materialize candidate diagnostics; installed fitting should drain the iterator."""
    return list(iter_candidates(method, prompt, portfolio))


def observed_union(outcomes):
    return {
        "kind": "DIAGNOSTIC_ONLY",
        "executable": False,
        "successes": sorted(set().union(*(set(v) for v in outcomes.values()))),
        "reason": "no_frozen_visible_only_portfolio_policy",
    }
