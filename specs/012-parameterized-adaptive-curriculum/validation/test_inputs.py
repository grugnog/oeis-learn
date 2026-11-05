"""Original-byte and explicit-search counterexamples; no workstation data needed."""
import importlib.util
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation.identity import digest,seal,write
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.structural import original_cells


def helper():
    spec=importlib.util.spec_from_file_location('pac_inputs',Path(__file__).with_name('inputs.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(root):
    refs=[]
    for cell in original_cells():
        path=root/(cell['name']+'.json');write(path,{'protected':'not parsed by host inventory'})
        refs.append(dict(cell,ref=reference(path,root)))
    receipt=seal(dict(version='generalization-command-result/v2',status='PASS',inventory=[r['ref'] for r in refs]))
    write(root/'receipt.json',receipt)
    return refs,reference(root/'receipt.json',root)


def test_exact_original_inventory_rejects_missing_corruption_and_failed_producer(tmp_path):
    h=helper();cells,receipt=fixture(tmp_path)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    result=h.inventory(tmp_path,cells,receipt)
    assert result['status']=='PASS' and len(result['originals'])==16
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError,match='16'):h.inventory(tmp_path,cells[:-1],receipt)
    (tmp_path/cells[0]['ref']['path']).write_text('changed')
    with pytest.raises(ValueError):h.inventory(tmp_path,cells,receipt)
    cells,receipt=fixture(tmp_path)
    write(tmp_path/'receipt.json',seal(dict(version='generalization-command-result/v2',status='FAIL',inventory=[r['ref'] for r in cells])))
    receipt=reference(tmp_path/'receipt.json',tmp_path)
    old=(tmp_path/'receipt.json').read_bytes()
    with pytest.raises(ValueError,match='original PASS'):h.inventory(tmp_path,cells,receipt)
    assert (tmp_path/'receipt.json').read_bytes()==old


def test_bounded_discovery_names_explicit_root_remedy_and_rejects_links(tmp_path):
    h=helper()
    for i in range(8):(tmp_path/str(i)).mkdir()
    with pytest.raises(ValueError,match='explicit.*root|inventory'):h.discover(tmp_path,'missing.json',max_entries=3)
    (tmp_path/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):h.discover(tmp_path,'missing.json',max_entries=30)


def test_weights_only_control_never_invents_match(tmp_path):
    h=helper();proof={k:digest(k) for k in ('initializer','order','model','optimizer','exposure','diagnostics','backend')}
    kwargs=dict(original_study=digest('study'),source=digest('source'),image=digest('image'),checkpoint=digest('weights'))
    matched=h.audit_control(proof,proof,**kwargs)
    assert matched['disposition']=='MATCHED'
    partial=h.audit_control({'initializer':proof['initializer']},proof,**kwargs)
    assert partial['disposition']=='REFERENCE_ONLY' and 'order' in partial['reasons']
    assert partial['historical_measurements'] is None
