import importlib.util
from pathlib import Path
import pytest
from tests.unit.test_generalization_metrics import pac_roster
from tests.unit.test_generalization_execution import pac_setup


def helper():
    spec=importlib.util.spec_from_file_location('pac_analysis',Path(__file__).with_name('analysis.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scoped_macro_completion_optional_attribution_and_stable_cases():
    h=helper();roster=pac_roster();exp,control,profiles,_=pac_setup()
    b=dict.fromkeys(roster,False);c=dict(b)
    b['certified/0/0']=True
    for key in ('certified/2/0',):c[key]=True
    result=h.analyze(roster,{'B':b,'C':c},experiment=exp,control=control,profiles=profiles)
    assert result['comparisons']['C-B']['interpretable']
    assert 'B-A' not in result['comparisons'] and result['reference_only_A']
    assert len(result['cases'])>=12 and result['cases']==h.analyze(roster,{'C':c,'B':b},experiment=exp,control=control,profiles=profiles)['cases']
    assert result['automatic_winner'] is False and 'decode_timeout' in result['missing_case_categories']
    certified=[k for k,v in roster.items() if v['category']=='certified']
    for key in certified[:4]:c[key]=None
    assert not h.analyze(roster,{'B':b,'C':c},experiment=exp,control=control,profiles=profiles)['comparisons']['C-B']['interpretable']
    c[certified[3]]=False
    assert h.analyze(roster,{'B':b,'C':c},experiment=exp,control=control,profiles=profiles)['comparisons']['C-B']['interpretable']


def test_unknown_scope_and_private_source_case_rejected_or_omitted():
    h=helper();exp,control,profiles,_=pac_setup();roster=pac_roster()
    with pytest.raises(ValueError):h.analyze(dict(list(roster.items())[:-1]),{},experiment=exp,control=control,profiles=profiles)
    with pytest.raises(ValueError):h.analyze(roster,{'D':{}},experiment=exp,control=control,profiles=profiles)


def test_generated_source_projection_excludes_hidden_fields_and_names_limitations():
    h=helper();roster=pac_roster();exp,control,profiles,_=pac_setup();outcomes={'B':dict.fromkeys(roster,False),'C':dict.fromkeys(roster,False)}
    records={a:{k:dict(selections={},details={'correct':{'details':[{'source':'(program (return (int 0)))'}]}},hidden_values=['secret'],target_source='private teacher') for k in roster} for a in outcomes}
    result=h.analyze(roster,outcomes,experiment=exp,control=control,profiles=profiles,records=records)
    assert len(result['cases'])>=12 and all(c['original_source'] for c in result['cases'])
    assert 'secret' not in str(result) and 'private teacher' not in str(result)
    markdown=h.case_markdown(result)
    assert markdown.count('## ')>=12 and 'solver limitation unresolved' in markdown
