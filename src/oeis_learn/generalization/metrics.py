"""Exact completion bounds, distinct from uncertainty across independent seeds."""
from dataclasses import dataclass
from fractions import Fraction as F
import random

@dataclass(frozen=True)
class Summary:
    n: int
    successes: int
    unresolved: int
    lower: F
    upper: F
    completion: F
    macro_lower: F
    macro_upper: F


def summarize(roster, outcomes):
    if not roster or set(outcomes) - set(roster) or any(v is not True and v is not False and v is not None for v in outcomes.values()):
        raise ValueError('outcomes must match frozen nonempty roster')
    groups = {}
    for row, family in roster.items():
        groups.setdefault(family, []).append(outcomes.get(row))
    n=len(roster); s=sum(v is True for v in outcomes.values()); u=sum(outcomes.get(k) is None for k in roster)
    lower=sum((F(sum(v is True for v in xs),len(xs)) for xs in groups.values()), F()) / len(groups)
    upper=sum((F(sum(v is not False for v in xs),len(xs)) for xs in groups.values()), F()) / len(groups)
    return Summary(n,s,u,F(s,n),F(s+u,n),F(n-u,n),lower,upper)


def paired_difference(roster, baseline, treatment, *, macro=False):
    a=summarize(roster,baseline); b=summarize(roster,treatment)
    return (b.macro_lower-a.macro_upper, b.macro_upper-a.macro_lower) if macro else (b.lower-a.upper,b.upper-a.lower)


def oracle_union(roster, runs):
    for run in runs: summarize(roster,run)
    merged={k: True if any(r.get(k) is True for r in runs) else False if runs and all(r.get(k) is False for r in runs) else None for k in roster}
    return {'scope':'scorer-only coverage', 'summary':summarize(roster,merged)}


def seed_uncertainty(paired_seeds, *, seed=0, draws=2000):
    """Percentile bootstrap of paired seed differences; never treats rows as seeds.

    Small seed counts are disclosed; censoring bounds must be reported separately.
    """
    if len(paired_seeds)<3 or not 100 <= draws <= 10000:
        raise ValueError('at least three paired seeds and bounded draws required')
    diffs=[F(b)-F(a) for a,b in paired_seeds.values()]
    rng=random.Random(seed)
    means=sorted(sum((rng.choice(diffs) for _ in diffs),F())/len(diffs) for _ in range(draws))
    return dict(unit='paired_seed',seeds=len(diffs),method='percentile-bootstrap-95',
                lower=means[int(draws*.025)],upper=means[min(draws-1,int(draws*.975))])
