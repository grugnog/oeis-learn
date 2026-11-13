"""Retirement is a storage operation, never permission to claim HIP results."""
import pytest

from oeis_learn.generalization.progressive.contracts import digest,load,reference,seal
from oeis_learn.pipeline_validation import progressive_calibration_storage as storage


@pytest.fixture
def measured(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_qualification import calibrate_iteration
    fixture(tmp_path/'inputs',interrupt=False)
    package=load(tmp_path/'inputs/training-input.json')
    producer=calibrate_iteration(package,tmp_path/'case',source=digest('cpu-only'),
        image=digest('image'),device='cpu',checkpoint_bytes=64<<20,seconds=60)
    return tmp_path/'case',producer


def test_retains_original_cpu_producer_and_full_hash_inventory_before_retiring(measured):
    root,producer=measured
    original=(root/'producer.json').read_bytes()
    before={str(p.relative_to(root)):reference(root,p) for p in (root/'checkpoints').rglob('*') if p.is_file()}
    ref=storage.retire(root,producer,maximum_bytes=128<<20)
    result=storage.verify(root,ref,producer)
    assert {r['path']:r for r in result['files']}==before
    assert result['device']=='cpu' and not (root/'checkpoints').exists()
    assert (root/'producer.json').read_bytes()==original
    assert load(root/'checkpoint-retirement-intent.json')['status']=='INVENTORIED'
    from oeis_learn.generalization.progressive.iteration import measured_producer
    with pytest.raises(ValueError,match='current exact'):
        measured_producer(producer,source=producer['source'],images=[producer['image']],compatibility=producer['compatibility'])


def test_failed_or_unbounded_calibration_retains_snapshots(measured):
    root,producer=measured
    with pytest.raises(ValueError,match='successful exact-input'):
        storage.retire(root,seal(dict(producer,status='FAIL')),maximum_bytes=128<<20)
    with pytest.raises(ValueError,match='storage bound'):
        storage.retire(root,producer,maximum_bytes=1)
    assert (root/'checkpoints').exists()
    assert not (root/'checkpoint-retirement-intent.json').exists()


def test_cleanup_failure_keeps_intent_and_never_publishes_completed_retirement(measured,monkeypatch):
    root,producer=measured
    def failed(path):
        assert load(root/'checkpoint-retirement-intent.json')['status']=='INVENTORIED'
        raise OSError('test cleanup failure')
    monkeypatch.setattr(storage.shutil,'rmtree',failed)
    with pytest.raises(OSError,match='cleanup failure'):
        storage.retire(root,producer,maximum_bytes=128<<20)
    assert (root/'checkpoints').exists()
    assert not (root/'checkpoint-retirement.json').exists()


def test_retirement_rejects_snapshot_alias_without_touching_target(measured,tmp_path):
    root,producer=measured
    target=tmp_path/'outside';target.write_bytes(b'keep')
    (root/'checkpoints/alias').symlink_to(target)
    with pytest.raises(ValueError):
        storage.retire(root,producer,maximum_bytes=128<<20)
    assert target.read_bytes()==b'keep' and (root/'checkpoints').exists()


def test_rolling_storage_counts_all_inputs_and_metadata_but_two_live_generations():
    cases=1296;checkpoint=4<<30;inputs=48<<30
    rolling=storage.required_bytes(cases,checkpoint,inputs,retention=storage.POLICY)
    retained=storage.required_bytes(cases,checkpoint,inputs)
    assert rolling==2*checkpoint+inputs+(16<<20)+cases*storage.METADATA_PER_CASE
    assert retained==2*cases*checkpoint+inputs+(16<<20)
    assert rolling<64<<30 and retained>10*(1<<40)
    with pytest.raises(ValueError,match='retention policy'):
        storage.required_bytes(cases,checkpoint,inputs,retention='delete-anything')


def test_resource_estimate_accounts_for_hourly_snapshots_and_legacy_policy():
    from oeis_learn.pipeline_validation.progressive_research_resources import estimate_training,policy
    from oeis_learn.pipeline_validation.research_limits import padded_us
    package=dict(shards=[[dict(tokens=[1])]],update_policy=dict(effective_batch=16))
    measured=dict(update_us=1000,checkpoint_us=1000000,probe_us=500)
    legacy=estimate_training(package,measured,delta=16000,target=16000,available=7200*1000000)
    hourly=estimate_training(dict(package,checkpoint_policy='hourly/v1'),measured,delta=16000,target=16000,available=7200*1000000)
    assert legacy['maximum_updates']==hourly['maximum_updates']==1000
    assert legacy['maximum_snapshots']==1000 and hourly['maximum_snapshots']==4
    assert hourly['estimated_work_us']==padded_us(1000000+4000000+125000)
    assert hourly['estimated_work_us']<legacy['estimated_work_us']/100
    assert policy(legacy=True)['version'].endswith('/v1') and policy(global_storage=True)['version'].endswith('/v2')
    assert policy()['version'].endswith('/v3')
