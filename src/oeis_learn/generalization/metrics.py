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


def _pac_roster(roster):
    from collections import Counter
    if len(roster)!=128 or any(set(r)!={'family','category','stratum'} or r['stratum']!='structural' or not isinstance(r['family'],str) or not r['family'] for r in roster.values()):
        raise ValueError('frozen 128-task structural roster required')
    counts=Counter(r['category'] for r in roster.values())
    families={c:{r['family'] for r in roster.values() if r['category']==c} for c in ('certified','ambiguous','unresolved')}
    if counts!=dict(certified=68,ambiguous=26,unresolved=34) or [len(families[c]) for c in families]!=[18,4,5] or len(set.union(*families.values()))!=27:
        raise ValueError('fixed structural category/family allocation')
    return families


def pac_structural_metrics(roster,outcomes):
    """All frozen families remain in each denominator, including censored tasks."""
    _pac_roster(roster)
    summarize({k:r['family'] for k,r in roster.items()},outcomes)
    results={}
    for category in ('certified','ambiguous','unresolved'):
        selected={k:r['family'] for k,r in roster.items() if r['category']==category}
        s=summarize(selected,{k:outcomes[k] for k in selected if k in outcomes})
        results[category]=dict(tasks=s.n,families=len(set(selected.values())),successes=s.successes,
                               complete=s.n-s.unresolved,censored=s.unresolved,
                               task_lower=str(s.lower),task_upper=str(s.upper),family_macro_lower=str(s.macro_lower),
                               family_macro_upper=str(s.macro_upper),completion=str(s.completion))
    return results


def pac_paired_counts(roster,baseline,treatment):
    _pac_roster(roster)
    for outcome in (baseline,treatment):summarize({k:r['family'] for k,r in roster.items()},outcome)
    selected=[k for k,r in roster.items() if r['category']=='certified']
    pairs=[(baseline.get(k),treatment.get(k)) for k in selected]
    complete=[(a,b) for a,b in pairs if a is not None and b is not None]
    return dict(tasks=len(selected),joint_complete=len(complete),both=sum(a and b for a,b in complete),
                baseline_only=sum(a and not b for a,b in complete),treatment_only=sum(b and not a for a,b in complete),
                neither=sum(not a and not b for a,b in complete),censored_pairs=len(pairs)-len(complete))
