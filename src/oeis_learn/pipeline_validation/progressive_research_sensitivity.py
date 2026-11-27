"""Collect independent, outcome-blind sensitivity cases without resetting costs."""
from collections import Counter
from pathlib import Path
import time
import traceback

from .progressive_workstation_state import write
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.sensitivity import validate_design, repeat_verdict


def run(design, output, *, seconds=7080):
    design = validate_design(design)
    output = Path(output)
    path = output / 'case-outcomes.json'
    if path.exists():
        raise ValueError('original sensitivity cases retained; no automatic budget reset')
    began = time.monotonic()
    cases, results = [], []
    def collect():
        write(path, seal(dict(version='progressive-sensitivity-cases/v1',
            design=digest(design), elapsed_seconds=time.monotonic()-began,
            requested=len(design['scenarios'])*design['repeats'], cases=cases)))
    collect()
    try:
        for ordinal, scenario in enumerate(design['scenarios']):
            counts = Counter()
            for repeat in range(design['repeats']):
                case = dict(scenario=ordinal, repeat=repeat, status='UNATTEMPTED')
                if time.monotonic()-began < seconds:
                    # Record dispatch before potentially expensive bootstrap work.
                    case['status'] = 'RUNNING'
                    cases.append(case); collect()
                    try:
                        verdict = repeat_verdict(design, ordinal, repeat)
                        case.update(status='COMPLETE', verdict=verdict)
                        counts[verdict] += 1
                    except Exception as exc:
                        case.update(status='FAILED', reason=type(exc).__name__+': '+str(exc),
                                    traceback=traceback.format_exc())
                else:
                    cases.append(case)
                collect()
            complete = sum(counts.values()) == design['repeats']
            promoted = counts['CONFIRMED_GAIN'] + counts['CONFIRMED_EFFICIENCY']
            positive = scenario['claim']=='efficiency' or scenario['effect_ppm'] >= 20000
            results.append(dict(assumptions=scenario, complete=complete,
                verdict_counts=dict(sorted(counts.items())),
                false_advancement_rate=promoted/design['repeats'] if complete and not positive else None,
                missed_gain_rate=1-promoted/design['repeats'] if complete and positive else None,
                inconclusive_rate=counts['INCONCLUSIVE']/design['repeats'] if complete else None,
                incomplete_rate=counts['INCOMPLETE']/design['repeats'] if complete else None))
            print(f'Research sensitivity: {ordinal+1}/{len(design["scenarios"])} scenarios',flush=True)
        result=seal(dict(version='progressive-collected-sensitivity/v1',kind='DIAGNOSTIC_ONLY',
            status='COMPLETE' if all(r['complete'] for r in results) else 'INCOMPLETE',
            design=digest(design), no_confirmation_inputs=True, research_ready=False,
            elapsed_seconds=time.monotonic()-began, scenarios=results,
            cases=digest(cases), case_statuses=dict(Counter(c['status'] for c in cases)),
            interpretation='prospective synthetic scenarios, not measured model power; missing cases have no estimated rates'))
        write(output/'sensitivity.json',result)
        return result
    finally:
        collect()


def main():
    output=Path('/work/output')
    run(load(output/'design.json'),output)
