import copy,json
import pytest
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.generalization_preparation import reference


def test_acceptance_archive_reconciles_originals_and_rejects_stale_status(tmp_path,monkeypatch):
    from tests.unit.test_generalization_feature_acceptance import fixture
    from oeis_learn.generalization.acceptance_archive import verify_acceptance
    from oeis_learn.pipeline_validation.generalization_qualification_io import public_closure
    from pathlib import Path
    from tests.unit import test_generalization_feature_acceptance as feature_fixture
    from tests.unit.test_generalization_qualification import producer
    from oeis_learn.generalization.qualification import FACTS
    (tmp_path/'base.json').write_text('{}');base=reference(tmp_path/'base.json',tmp_path)
    def full_producer(root):
        original,_=producer(root)
        raw=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['qualification']
        def refs(value):
            if isinstance(value,dict):
                if set(value)=={'path','bytes','sha256'}:return base
                return {k:refs(v) for k,v in value.items()}
            if isinstance(value,list):return [refs(v) for v in value]
            return value
        facts={k:refs(raw[k]) for k in FACTS}
        # Match the canonical producer's actual pinned-build closure, not JSON
        # stand-ins that hide filename-policy failures during report packaging.
        builds=[]
        for name in ('Dockerfile','pyproject.toml','uv.lock','crates/oeis_synthesis_core/Cargo.lock'):
            path=root/'build'/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('pinned build fixture\n');builds.append(reference(path,root))
        facts['build_dependencies']=builds
        original=seal(dict(original,facts=facts,suites=raw['suite_selection'],capabilities=raw['covered_capabilities']))
        return original,facts
    monkeypatch.setattr(feature_fixture,'producer',full_producer)
    original,facts,feature=fixture(tmp_path)
    (tmp_path/'producer.json').write_text(json.dumps(original))
    authority=seal(dict(version='generalization-qualification/v2',suite_selection=original['suites'],covered_capabilities=original['capabilities'],
        **facts,producer=dict(receipt=reference(tmp_path/'producer.json',tmp_path),status='PASS',artifacts=original['artifacts']),artifact_inventory=public_closure(tmp_path,original),status='PASS'))
    (tmp_path/'qualification.json').write_text(json.dumps(authority));(tmp_path/'feature.json').write_text(json.dumps(feature))
    record=seal(dict(version='generalization-acceptance-archive/v1',session='session',qualification=reference(tmp_path/'qualification.json',tmp_path),
        feature=reference(tmp_path/'feature.json',tmp_path),cases=feature['cases'],software_status='PASS',scientific_status='UNMEASURED',omissions=[],workflow=None,
        inventory=[reference(p,tmp_path) for p in sorted(tmp_path.rglob('*')) if p.is_file()]))
    assert verify_acceptance(record,root=tmp_path)['software_status']=='PASS'
    changed=copy.deepcopy(record);changed['cases'].pop('refresh');changed=seal(changed)
    with pytest.raises(ValueError):verify_acceptance(changed,root=tmp_path)
    changed=copy.deepcopy(record);changed['software_status']='PARTIAL';changed=seal(changed)
    with pytest.raises(ValueError):verify_acceptance(changed,root=tmp_path)
    from oeis_learn.pipeline_validation.generalization_reporting import package_acceptance
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    full_archive=package_acceptance(tmp_path,record,output=tmp_path/'complete')
    assert verify_archive(full_archive)['feature_qualified'] is True
    partial=copy.deepcopy(record);partial['feature']=None;partial['cases'].pop('refresh')
    partial.update(software_status='PARTIAL',omissions=['refresh']);partial=seal(partial)
    partial_archive=package_acceptance(tmp_path,partial,output=tmp_path/'partial')
    verified=verify_archive(partial_archive)
    assert verified['integrity']=='PASS' and verified['software_status']=='PARTIAL' and verified['feature_qualified'] is False


def test_pinned_build_paths_are_public_without_admitting_private_or_arbitrary_files():
    from pathlib import Path
    from oeis_learn.pipeline_validation.bundle import public
    for prefix in ('','payloads/'):
        for name in ('build/Dockerfile','build/pyproject.toml','build/uv.lock','build/crates/core/Cargo.lock'):
            assert public(Path(prefix+name))
        for name in ('private/build/Dockerfile','build/.env','build/weights.pt','build/source.py',
                     'build/arbitrary.lock','other/Dockerfile','build/private/Cargo.lock',
                     'build/crates/private/Cargo.lock','inputs/build/Dockerfile'):
            assert not public(Path(prefix+name))
