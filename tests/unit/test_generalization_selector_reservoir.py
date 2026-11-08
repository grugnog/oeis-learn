"""The adopted reservoir changes only mechanical prefix lengths and custody."""
import ast
import copy
import hashlib
from pathlib import Path

import pytest

from oeis_learn.pipeline_validation.identity import seal

ROOT=Path(__file__).resolve().parents[2]


def test_amendment_preserves_scientific_fields_and_every_old_draw_prefix():
    from oeis_learn.generalization.selector_validation import fresh_design,fresh_draw
    from oeis_learn.generalization.selector_reservoir import reservoir_design,draw,registry
    old=fresh_design();new=reservoir_design();policy=registry()
    assert old['identity']=='sha256:f5b92efafe8394952de1ff1b00429288fbcf461f66aaeade44608ee9ee496441'
    assert sum(c['draws'] for c in new['cells'])==7664<=new['identity_capacity']
    for a,b in zip(old['cells'],new['cells']):
        assert dict(a,draws=policy['draws'].get(a['family'],a['draws']))==b
    changed={'version','identity','parent_protocol','reservoir_amendment','cells'}
    assert {k:v for k,v in old.items() if k not in changed}=={k:v for k,v in new.items() if k not in changed}
    original=fresh_draw(old);expanded=draw(new)
    for cell in old['cells']:
        a=[r for r in original if r['family']==cell['family']]
        b=[r for r in expanded if r['family']==cell['family']]
        assert b[:len(a)]==a
    for key,value in [('seed',old['seed']+1),('numeric_law','changed')]:
        with pytest.raises(ValueError):draw(seal(dict(new,**{key:value})))
    with pytest.raises(ValueError):fresh_draw(new)


def source_pair():
    from oeis_learn.generalization.selector_reservoir import registry
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    policy=registry();current=collect_snapshot(ROOT);old=dict(current)
    for p in policy['new_files']:old.pop(p,None)
    for p,r in policy['files'].items():old[p]=r['sha256']
    old.pop('scripts/pac_fresh_support_audit.py',None)
    return old,current


def test_exact_source_compatibility_preserves_scoring_and_native_closure():
    from oeis_learn.generalization.selector_reservoir import source_compatibility
    old,current=source_pair();proof=source_compatibility(ROOT,old,current)
    assert proof==seal(proof) and proof['version']=='pac-selector-probe-compatibility/v1'
    for name in ['src/oeis_learn/generalization/selector_validation.py','crates/oeis_synthesis_core/src/fit.rs','Dockerfile']:
        changed=dict(current);changed[name]='sha256:'+'0'*64
        with pytest.raises(ValueError,match='protected source'):source_compatibility(ROOT,old,changed)


@pytest.mark.parametrize('mutation',['score','global','decorator','signature'])
def test_shared_module_cannot_smuggle_inference_or_import_side_effects(mutation):
    from oeis_learn.generalization.selector_reservoir import source_compatibility
    old,current=source_pair();name='src/oeis_learn/pipeline_validation/selector_validation_roles.py'
    text=(ROOT/name).read_text()
    if mutation=='score':text=text.replace('def score(', 'def score_changed(')
    elif mutation=='global':text+='\nCONTROLS=()\n'
    elif mutation=='decorator':text=text.replace('def publish_structural(', '@print("bad")\ndef publish_structural(')
    else:text=text.replace('def publish_structural(inputs, output, settings, *, seconds):','def publish_structural(inputs, output, settings, *, seconds=1):')
    changed=dict(current);changed[name]='sha256:'+hashlib.sha256(text.encode()).hexdigest()
    reader=lambda path:text if path==name else (ROOT/path).read_text()
    with pytest.raises(ValueError,match='protected|signature|decorator'):
        source_compatibility(ROOT,old,changed,text_reader=reader)


@pytest.mark.parametrize('mutation',['metric','attachment'])
def test_report_exception_allows_only_the_exact_public_metadata_member(mutation):
    from oeis_learn.generalization.selector_reservoir import source_compatibility
    old,current=source_pair();name='src/oeis_learn/pipeline_validation/selector_validation_commands.py'
    text=(ROOT/name).read_text()
    if mutation=='metric':text=text.replace('floor=math.ceil(.95*n)','floor=1')
    else:text=text.replace("archive.write(output/'admission.json','admission.json')","archive.write(output/'private/targets.json','admission.json')")
    changed=dict(current);changed[name]='sha256:'+hashlib.sha256(text.encode()).hexdigest()
    reader=lambda path:text if path==name else (ROOT/path).read_text()
    with pytest.raises(ValueError,match='protected'):source_compatibility(ROOT,old,changed,text_reader=reader)
