"""Fixed visible-only baselines and shared portfolio accounting.

No observed outcome union is an executable policy. Model loading delegates to
existing constructors/loaders; conventional methods disclose their narrow priors.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import random
import time
from .contracts import verify_refs, digest
from .cohorts import validate_prompt, validate_profile, DEFAULT_PROFILE

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
    from oeis_learn.fresh.bigint_worker import models
    from copy import deepcopy
    from oeis_learn.fresh.bigint_config import resolve

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
    for p in prompts:
        validate_prompt(p)
    if mode == "absent":
        return [
            dict(id=p["id"], indices=list(p["indices"]), values=None, conditioning="absent_marker")
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
            dict(id=p["id"], indices=list(p["indices"]), values=list(prompts[donors[i]]["values"]))
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

    def __post_init__(self):
        validate_profile(self.profile)

    @property
    def deadline(self):
        return self.started + self.profile["portfolio_us"] / 1_000_000

    def reserve(self, method, *, now=None):
        if method not in METHODS:
            raise ValueError("unregistered baseline")
        now = time.monotonic() if now is None else now
        if now >= self.deadline or self.attempts >= self.profile["attempts"]:
            raise TimeoutError("total portfolio allowance exhausted")
        self.attempts += 1
        return min(self.deadline, now + self.profile["fit_us"] / 1_000_000)

    def observe(self, method, *, candidate, started, ended, outcome):
        if ended < started:
            raise ValueError("invalid measured interval")
        self.costs.append(
            dict(
                method=method,
                candidate=candidate,
                elapsed_us=int((ended - started) * 1_000_000),
                outcome=outcome,
            )
        )

    def report(self):
        methods = {}
        for cost in self.costs:
            row = methods.setdefault(cost["method"], dict(candidates=0, elapsed_us=0))
            row["candidates"] += int(cost["candidate"] is not None)
            row["elapsed_us"] += cost["elapsed_us"]
        return dict(
            attempts=self.attempts,
            methods=methods,
            details=list(self.costs),
            total_elapsed_us=max(0, int((time.monotonic() - self.started) * 1_000_000)),
            accounting="one shared portfolio; overlapping method timings are not added to wall time",
        )


def service_outcome(result, *, observed_at, deadline, fitting_supported=True):
    if observed_at > deadline:
        return dict(success=False, complete=False, reason="expired_service_result")
    if not fitting_supported:
        return dict(success=False, complete=False, reason="unsupported_fitting")
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


def polynomial(prompt, *, max_degree=3):
    validate_prompt(prompt)
    if type(max_degree) is not int or not 0 <= max_degree <= 3:
        raise ValueError("fixed polynomial applicability degree <=3")
    xs = [int(x) for x in prompt["indices"]]
    ys = [int(x) for x in prompt["values"]]
    for degree in range(max_degree + 1):
        coefficients = _solve(
            [[x**j for j in range(degree + 1)] for x in xs[: degree + 1]], ys[: degree + 1]
        )
        if coefficients is None or any(c.denominator != 1 for c in coefficients):
            continue
        if any(sum(c * x**j for j, c in enumerate(coefficients)) != y for x, y in zip(xs, ys)):
            continue
        source = f"(int {int(coefficients[-1])})"
        for coefficient in reversed(coefficients[:-1]):
            source = f"(add (mul {source} $index) (int {int(coefficient)}))"
        return "(program (return " + source + "))"
    return None


def linear_recurrence(prompt, *, max_order=2):
    validate_prompt(prompt)
    if type(max_order) is not int or not 1 <= max_order <= 2:
        raise ValueError("fixed recurrence order <=2")
    if prompt["indices"] != [str(i) for i in range(20)]:
        return None
    y = [int(v) for v in prompt["values"]]
    for order in range(1, max_order + 1):
        coeff = _solve(
            [[y[n - j - 1] for j in range(order)] for n in range(order, 2 * order)],
            y[order : 2 * order],
        )
        if coeff is None or any(c.denominator != 1 for c in coeff):
            continue
        if any(
            sum(coeff[j] * y[n - j - 1] for j in range(order)) != y[n] for n in range(order, 20)
        ):
            continue
        if order == 1:
            return f"(program (local $b0 bigint) (local $c0 control) (set $b0 (int {y[0]})) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (int {int(coeff[0])})))) (return (get $b0)))"
        return f"(program (local $b0 bigint) (local $b1 bigint) (local $b2 bigint) (local $c0 control) (set $b0 (int {y[0]})) (set $b1 (int {y[1]})) (loop $c0 (to_control $index) (set $b2 (add (mul (int {int(coeff[0])}) (get $b1)) (mul (int {int(coeff[1])}) (get $b0)))) (set $b0 (get $b1)) (set $b1 (get $b2))) (return (get $b0)))"
    return None


def candidates(method, prompt, portfolio, *, max_degree=3, max_order=2):
    """Concrete conventional or parameterized generic sources with charged attempts."""
    validate_prompt(prompt)
    if method == "templates":
        from oeis_learn.fresh.template_baseline import template_bank

        sources = template_bank()["sources"]
    elif method in {"polynomial", "linear_recurrence"}:
        sources = [None]
    else:
        raise ValueError("model candidates require the installed generator")
    output = []
    for source in sources:
        began = time.monotonic()
        try:
            deadline = portfolio.reserve(method)
        except TimeoutError:
            break
        if method == "polynomial":
            source = polynomial(prompt, max_degree=max_degree)
        elif method == "linear_recurrence":
            source = linear_recurrence(prompt, max_order=max_order)
        ended = time.monotonic()
        expired = ended > portfolio.deadline
        portfolio.observe(
            method,
            candidate=digest(source) if source is not None else None,
            started=began,
            ended=ended,
            outcome="deadline" if expired else "candidate_generated" if source else "inapplicable",
        )
        if expired:
            break
        output.append(
            dict(method=method, source=source, fit_deadline=deadline, applicable=source is not None)
        )
    return output


def observed_union(outcomes):
    return dict(
        kind="DIAGNOSTIC_ONLY",
        executable=False,
        successes=sorted(set().union(*(set(v) for v in outcomes.values()))),
        reason="no_frozen_visible_only_portfolio_policy",
    )
