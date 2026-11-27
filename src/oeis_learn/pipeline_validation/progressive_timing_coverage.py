"""Cost coverage for training-only calibration, separate from scientific completion.

An exhausted decoder token allowance is a measured terminal attempt. It is not
a completed program or a successful scientific observation. Work skipped because
no program was produced needs an explicit conservative reserve, not a zero-cost
extrapolation to later trained checkpoints.
"""
from .identity import seal


def coverage(generated, scored):
    from .pilot_clocks import DRAIN, clock_record
    methods = generated['methods']
    neural = [m for m in methods if m in {'correct', 'shuffled', 'absent', 'template'}]
    clocks = generated.get('clock', {}).get('clocks', [])
    expected = {(member, mode) for member in generated['records'] for mode in neural}
    actual = [(c.get('prompt'), c.get('mode')) for c in clocks]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('selected timing portfolio clock roster incomplete')
    if generated.get('clock', {}).get('policy') != DRAIN:
        raise ValueError('original drain timing clock policy required')
    for clock in clocks:
        replay=clock_record(policy=DRAIN,**{k:clock[k] for k in (
            'prompt','mode','snapshot','start','scope_deadline','finish','expected','started','terminal')})
        if clock != replay:
            raise ValueError('original timing portfolio clock differs')
    if any(c.get('cost_complete') is not True or c.get('expected') != 8
           or c.get('started') != 8 or c.get('terminal') != 8
           or c.get('termination') != 'attempt_roster_terminal' for c in clocks):
        raise ValueError('selected timing attempt roster did not terminate')
    if set(scored['outcomes']) != set(generated['records']):
        raise ValueError('timing scoring member roster incomplete')
    token_capped = native_reserve = scoring_reserve = 0
    selected = ['raw', 'rescue', 'equal_budget'] if 'correct' in methods else []
    selected += [m for m in methods if m != 'correct']
    unresolved = 0
    for member, row in generated['records'].items():
        for mode in neural:
            details = row['costs'][mode]['details']
            if len(details) != 8 or [d['position'] for d in details] != list(range(8)):
                raise ValueError('timing attempt detail roster incomplete')
            for d in details:
                generation = d['generation']
                if generation.get('complete') is not True:
                    if (generation.get('reason') != 'token_cap_without_eos'
                            or len(d.get('tokens', [])) != 1024
                            or d['fitting'].get('outcome') != 'not_attempted'):
                        raise ValueError('timing generation failed outside the registered token allowance')
                    token_capped += 1
                    # Every later decoded attempt may use its four-second fit;
                    # the first neural draw also has a separate raw service.
                    native_reserve += 4_000_000
                    if d['position'] == 0 and mode != 'template':
                        native_reserve += 4_000_000
                elif d['fitting'].get('complete') is not True:
                    raise ValueError('timing native fitting incomplete')
        for mode in methods:
            if mode in {'polynomial', 'linear_recurrence'} and row['arm_evidence'][mode].get('complete') is not True:
                raise ValueError('timing conventional service incomplete')
        for method in selected:
            outcome = scored['outcomes'][member][method]
            unresolved += outcome.get('complete') is not True
            phase = outcome.get('scoring', {})
            if row['selections'][method] is not None:
                if phase.get('complete') is not True:
                    raise ValueError('timing selected candidate scoring incomplete')
            else:
                if phase.get('outcome') != 'no_selected_candidate':
                    raise ValueError('timing absent selection lacks terminal scorer evidence')
                # Native and reference full-horizon execution each have 30 s.
                # No candidate means this service path was not measured at all.
                scoring_reserve += 60_000_000
    return seal(dict(version='progressive-timing-coverage/v1',
        selected_portfolios=len(expected), terminal_attempts=8*len(expected),
        token_capped_attempts=token_capped, unresolved_scientific_outcomes=unresolved,
        unmeasured_native_reserve_us=native_reserve,
        unmeasured_scoring_reserve_us=scoring_reserve,
        interpretation='cost coverage only; original scientific completion and censoring unchanged',
        reserve_policy='unexercised fit/raw: 4 s each; unexercised full scoring: 30 s per executor'))
