"""Owning caller facade for the single native fitter and independent admission.

Retained scalar/wrapping/recurrence skeletons are explicitly unsupported, never
translated into a different numerical profile. No parameters are searched here.
"""

from dataclasses import dataclass
from contextlib import nullcontext
import time
from .admission import VisiblePrompt
from .contracts import identity


@dataclass(frozen=True)
class FitResult:
    outcome: str
    source: str
    original_source: str
    profile: str
    prompt: VisiblePrompt
    response: dict
    elapsed_ms: float
    reason: str | None = None
    policy: str = "none"

    @property
    def constants(self):
        admitted = self.response.get("admitted", [])
        return tuple(int(v) for v in admitted[0]["assignment"].values()) if admitted else ()

    def to_qualified(self):
        from oeis_learn.experiments.qualified_models import seal_record

        return seal_record(
            "native_grounding_result",
            original_source=self.original_source,
            profile=self.profile,
            indices=list(self.prompt.indices),
            values=list(self.prompt.values),
            result=self.response,
            policy=self.policy,
        )


def fit_program(
    source,
    terms,
    *,
    indices=None,
    profile="wat_i256_checked_v1",
    timeout_ms=4000,
    max_parameters=8,
    client=None,
    policy=None,
):
    from .client import SynthesisClient

    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
        raise ValueError("finite native fitting deadline required")
    if max_parameters != 8:
        raise ValueError(
            "native-fitting/v1 has a frozen eight-parameter cap; use the bigint recipe"
        )
    visible = tuple(str(v) for v in terms[:20])
    ns = tuple(str(i) for i in (range(len(visible)) if indices is None else indices[:20]))
    prompt = VisiblePrompt(ns, visible)
    started = time.monotonic()
    if (
        profile not in {"wat_i256_checked_v1", "bigint-body/v1"}
        or "const_?" in source
        or source.lstrip().startswith("(module")
    ):
        reason = "retired scalar/placeholder fitting; use bigint-body/v1 with explicit param.get or the checked i256 concrete subset"
        return FitResult(
            "unsupported",
            source,
            source,
            profile,
            prompt,
            dict(outcome="unsupported", admitted=[], reason=reason),
            0.0,
            reason,
        )
    policy = policy or ("none" if profile == "bigint-body/v1" else "eligible-literals/v1")
    contract = identity(
        dict(
            api="native-fit-call/v1",
            profile=profile,
            source=source,
            prompt=prompt.sha256,
            policy=policy,
            timeout_ms=timeout_ms,
        )
    )
    with nullcontext(client) if client is not None else SynthesisClient(contract) as native:
        response = native.fit(
            source, prompt, profile=profile, policy=policy, deadline=started + timeout_ms / 1000
        )
    admitted = response.get("admitted", [])
    return FitResult(
        response["outcome"],
        admitted[0]["source"] if admitted else source,
        source,
        profile,
        prompt,
        response,
        (time.monotonic() - started) * 1000,
        response.get("reason", (response.get("native") or {}).get("reason")),
        policy,
    )


def require_native_solver_settings(config):
    """Retained diagnostic recipes must never silently consume retired switches."""
    settings = config.get("solver")
    if settings is None:
        return
    fields = {"profile", "coefficient_bits", "timeout_ms", "max_parameters"}
    if (
        not isinstance(settings, dict)
        or set(settings) != fields
        or settings["profile"] != "native-fitting/v1"
        or settings["coefficient_bits"] != 4096
        or settings["max_parameters"] != 8
        or type(settings["timeout_ms"]) is not int
        or not 1 <= settings["timeout_ms"] <= 60000
    ):
        raise ValueError(
            "retired solver settings; use explicit native-fitting/v1 settings or configs/foundation/bigint_*.yaml"
        )
