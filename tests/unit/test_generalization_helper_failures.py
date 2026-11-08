"""Bounded installed helper diagnostics retain provenance without private text."""
import json
import pytest
from oeis_learn.pipeline_validation.identity import seal,write
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.pipeline_validation.generalization_helper_failures import capture_helper_failure,helper_diagnostic


def fixture(root,exc):
    public=capture_helper_failure(root,'baseline',exc)
    row=dict(stage='baseline',status=public['status'],artifacts=[reference(root/'helper-failure.json',root)])
    return public,row


@pytest.mark.parametrize('exception,status',[(FileNotFoundError,'FAIL'),(KeyboardInterrupt,'INTERRUPTED')])
def test_capture_retains_bounded_original_privately_and_validates_without_helper_import(tmp_path,exception,status):
    secret='/private/operator/'+('x'*70000)
    record,row=fixture(tmp_path,exception(secret))
    assert record['status']==status and record['location'] is None
    assert secret[:20] not in json.dumps(record)
    private=json.loads((tmp_path/'private/helper-exception.json').read_text())
    assert private['truncated'] is True and private['error']==secret[:65536]
    assert helper_diagnostic(tmp_path,row)==(record,None)


@pytest.mark.parametrize('mutation',['unbound','duplicate','hash','seal','stage','status','failure','kind','message','location','function','line'])
def test_unverified_or_resealed_unsafe_diagnostic_is_omitted(tmp_path,mutation):
    record,row=fixture(tmp_path,FileNotFoundError('private teacher payload'))
    path=tmp_path/'helper-failure.json'
    if mutation=='unbound':row['artifacts']=[]
    elif mutation=='duplicate':row['artifacts']*=2
    elif mutation=='hash':path.write_text('{}')
    else:
        if mutation=='seal':record['identity']='sha256:'+'0'*64
        elif mutation=='stage':record['stage']='admission'
        elif mutation=='status':record['status']='INTERRUPTED'
        elif mutation=='failure':record['failure']='private teacher payload'
        elif mutation=='kind':record['failure']=seal(dict(record['failure'],kind='/private/operator'))
        elif mutation=='message':record['message']='private teacher payload'
        elif mutation=='location':record['location']=['private teacher payload']
        elif mutation=='function':record['location']=dict(function='/private/operator',line=1)
        elif mutation=='line':record['location']=dict(function='execute_plan',line=True)
        if mutation!='seal':record=seal(record)
        write(path,record,exclusive=False)
        row['artifacts']=[reference(path,tmp_path)]
    value,omission=helper_diagnostic(tmp_path,row)
    assert value is None and omission=='unverified temporary helper diagnostic omitted: baseline'
    assert 'private teacher' not in omission and '/private' not in omission
