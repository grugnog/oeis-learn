"""Checkpoint/visible-input I/O around the Phase B evaluation policies.

Callbacks are supplied by the qualified model/native role. This module never
opens a scorer target or reconstructs unavailable weights from public evidence.
"""
from oeis_learn.generalization.contracts import verify_artifact,shape
from oeis_learn.generalization.proposals import portfolio
from oeis_learn.generalization.diagnostics import join_attempts
from .identity import confined,seal


def evaluate_checkpoint(root,checkpoint,prompts,*,load,generate,fit,controls=('raw','rescue')):
    path=confined(root,checkpoint['path'],exists=False)
    if not path.exists():
        return dict(availability='unavailable',checkpoint=checkpoint['sha256'],attempts=[],reason='MISSING_ARTIFACT')
    verify_artifact(root,checkpoint)
    model=load(path)
    attempts=[]
    for prompt in prompts:
        shape('visible-prompt',prompt)
        draws=[];records=[];fits={}
        for control in controls:
            for draw in generate(model,dict(prompt),control):
                draws.append(draw)
                portfolio(draws)
                id=f"{prompt['id']}:{len(draws)-1}"
                records.append(dict(id=id,prompt=prompt['id'],control=control,source=draw.get('source')))
                fits[id]=fit(draw.get('source'),dict(prompt))
        attempts.extend(join_attempts(records,fits))
    return seal(dict(version='generalization-checkpoint-evaluation/v1',availability='available',checkpoint=checkpoint['sha256'],attempts=attempts))
