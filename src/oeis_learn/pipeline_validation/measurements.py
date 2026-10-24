"""Finite reset paired execution, separating exploration from confirmation."""
from .comparisons import ORDER,decide
from .identity import seal, write


def paired(contract, output, *, reset, run, confirmation=False):
    """Callbacks are internal adapters, never commands supplied by a manifest.

    reset must restore full diagnostic state and isolated caches. run returns actual
    complete work and denominators; partial work is never converted to a speedup.
    """
    if not contract.get('identity') or not contract.get('calibration'):
        raise ValueError('sealed calibrated contract required before candidates')
    order=ORDER if confirmation else ['AB','BA']
    rows=[]
    reset('warmup',contract)
    run('A','warmup',contract)
    for ordinal,sequence in enumerate(order):
        pair={}
        for arm in sequence:
            reset(arm,contract)
            pair[arm]=run(arm,ordinal,contract)
        row=seal(dict(version='pipeline-pair/v1',contract=contract['identity'],ordinal=ordinal,
            kind='confirmation' if confirmation else 'exploratory',order=sequence,arms=pair))
        write(output/f'pair-{ordinal}.json',row)
        rows.append(row)
    return rows
