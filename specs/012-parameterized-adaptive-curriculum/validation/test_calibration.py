from pathlib import Path
import importlib.util
import pytest


def helper():
    spec=importlib.util.spec_from_file_location('pac_calibration',Path(__file__).with_name('calibration.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_complete_samples_raise_allowances_and_missing_never_means_zero():
    h=helper()
    samples={k:[2.,4.] for k in h.REQUIRED}
    value=h.forecast(samples,updates=10000,arm='D',existing=dict(learner=100000,generation=30000,preparation=100000,diagnostics=1))
    assert value['learner']['seconds']>=100000 and value['diagnostics']['counts']['reward_blocks']==1250
    assert value['samples']['cold_native']['count']==2 and value['samples']['cold_native']['worst_seconds']==4
    assert value['guaranteed_worst_case'] is False
    for k in h.REQUIRED:
        missing=dict(samples);missing[k]=[]
        with pytest.raises(ValueError):h.forecast(missing,updates=100,arm='C',existing={})


def test_actual_calibration_handler_refuses_cpu_fallback(tmp_path):
    from oeis_learn.pipeline_validation.generalization_pac_calibration import calibrate
    import torch
    if torch.version.hip:pytest.skip('CPU-only rejection fixture')
    with pytest.raises(RuntimeError):calibrate({},tmp_path,{},seconds=100)


def test_operational_rebind_preserves_science_and_only_raises_limits():
    from tests.unit.test_generalization_execution import pac_setup
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.generalization_execution import bind_pac,pac_budget_projection,POLICY_ID
    from oeis_learn.pipeline_validation.generalization_pac_calibration import apply_calibration
    from oeis_learn.pipeline_validation.identity import seal,digest
    exp,control,profiles,_=pac_setup();limits=envelope(purpose='research',arms=['B','C'])
    exp.update(backend=POLICY_ID,budget_graph=pac_budget_projection(limits))
    runtime=dict(version='pac-runtime/v1',study='',purpose='research',recipe=dict(path='recipe',bytes=1,sha256=digest('r')),
        source=dict(path='source',bytes=1,sha256=digest('s')),images=[digest('cpu'),digest('gpu')],control=control,profiles=profiles,
        preparation=None,prepared=None,envelope=limits,feature=None,engine='podman')
    exp,runtime=bind_pac(seal(exp),runtime)
    forecasts={a:helper().forecast({k:[1.] for k in helper().REQUIRED},updates=10000,arm=a,existing={}) for a in ['B','C']}
    record=seal(dict(version='pac-calibration/v1',status='PASS',source=exp['source'],environment=digest(dict(images=runtime['images'],engine='podman',backend=POLICY_ID)),forecasts=forecasts))
    new,side=apply_calibration(exp,runtime,record)
    assert all(new[k]==exp[k] for k in exp if k not in {'identity','execution','budget_graph'})
    assert side['envelope']['learner']['work']>=runtime['envelope']['learner']['work']
    bad=seal(dict(record,source=digest('foreign')))
    with pytest.raises(ValueError):apply_calibration(exp,runtime,bad)
