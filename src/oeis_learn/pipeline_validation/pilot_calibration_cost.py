"""Candidate full-clock costs; never accepted as legacy long-run authority."""
from .identity import seal
from .pilot_clocks import POLICY
from .pilot_cost import forecast as legacy_forecast


def forecast(observations,*,source,work,archives=()):
    # Reuse exact component validation/calculation, preserving the v1 evaluator.
    # Malformed/legacy candidates fail closed rather than manufacture clock evidence.
    try:legacy=legacy_forecast(observations,source=source,work=work,archives=archives)
    except (KeyError,TypeError,ValueError):legacy=legacy_forecast([],source=source,work=work,archives=[])
    clocks=all(r.get('evaluation_policy')==POLICY and r.get('clock_complete') is True and r.get('portfolios')==1184
        for r in observations) and len(observations)==3
    valid=clocks and legacy['projected_seconds'] is not None and legacy['archive_envelope_seconds'] is not None
    evaluation=legacy['evaluation_seconds'] if valid else None
    projected=legacy['projected_seconds'] if valid else None
    fits=valid and evaluation<=36000 and projected<=.8*172800 and legacy['archive_envelope_seconds']<=900
    return seal(dict(version='pipeline-pilot-calibration-cost/v1',source=source,work=work,evaluation_policy=POLICY,
        observations=observations,archives=list(archives),generator_role_seconds=legacy['generator_role_seconds'] if valid else None,
        scorer_role_seconds=legacy['scorer_role_seconds'] if valid else None,evaluation_seconds=evaluation,projected_seconds=projected,
        archive_envelope_seconds=legacy['archive_envelope_seconds'],evaluation_reservation_seconds=36000,safety_seconds=41400,
        inclusive_seconds=172800,within_budget=bool(fits),status='CANDIDATE_COST_ONLY' if fits else 'OVER_BUDGET' if valid else 'UNKNOWN',
        long_run_admitted=False,automatic_run_authorized=False,
        limitation='New policy candidate only; old long-run authority rejects it. Scientific incomplete outcomes stay unresolved.'))
