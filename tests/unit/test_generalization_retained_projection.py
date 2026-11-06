"""Retained reference projection is installed, exclusive and content verified."""
import pytest
from oeis_learn.pipeline_validation.identity import digest,seal
from oeis_learn.pipeline_validation.generalization_preparation import reference


@pytest.mark.parametrize('mode',['reference-only','fresh-control'])
def test_projection_is_consumable_and_rejects_stale_or_changed_originals(tmp_path,mode):
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import project_retained_checkpoint,checkpoint_request
    source=tmp_path/'original';source.mkdir();(source/'weights.pt').write_bytes(b'verified original')
    descriptor=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=reference(source/'weights.pt',source),
        state=None,initialization=digest('initial'),parent=None,tokens=25000000,updates=100,pins=['latest'],compatibility=digest('old control')))
    retained=dict(weights_root='original',checkpoint=descriptor)
    study=dict(control_mode=mode,arms=['A','B','C'] if mode=='fresh-control' else ['B','C'])
    runtime=dict(control=dict(checkpoint=descriptor['identity']))
    output=tmp_path/'new/private/retained'
    request=project_retained_checkpoint(tmp_path,retained,output,study,runtime)
    arm,root,actual=checkpoint_request(tmp_path,request,study,runtime,{}, {})
    assert arm==('A-reference' if mode=='fresh-control' else 'A')
    assert root==output and actual==descriptor
    assert (output/'weights.pt').read_bytes()==b'verified original'
    with pytest.raises(FileExistsError):project_retained_checkpoint(tmp_path,retained,output,study,runtime)
    (source/'weights.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError):project_retained_checkpoint(tmp_path,retained,tmp_path/'other',study,runtime)
    assert (output/'weights.pt').read_bytes()==b'verified original'
