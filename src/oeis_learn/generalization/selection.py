"""Selection-supervision-only checkpoint ordering."""
from fractions import Fraction
from .metrics import summarize
from .interfaces import SelectionResult


def select(candidates):
    eligible=[]; ids=set()
    for c in candidates:
        if set(c)!={'id','tokens','role','structural','parameter'} or c['role']!='selection' or c['id'] in ids:
            raise ValueError('selection-only unique candidate required')
        if type(c['tokens']) is not int or c['tokens']<0: raise ValueError('token exposure')
        ids.add(c['id']); structural=summarize(*c['structural']); parameter=summarize(*c['parameter'])
        if min(structural.completion,parameter.completion)<Fraction(95,100): continue
        eligible.append((-structural.macro_lower,-parameter.macro_lower,c['tokens'],c['id']))
    eligible.sort()
    return SelectionResult(eligible[0][-1] if eligible else None,tuple(c[-1] for c in eligible),'selected' if eligible else 'inconclusive_retain_baseline')
