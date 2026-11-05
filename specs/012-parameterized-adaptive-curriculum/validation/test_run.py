import importlib.util
from pathlib import Path
import pytest


def helper():
    spec=importlib.util.spec_from_file_location('pac_serial',Path(__file__).with_name('run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scope_order_gates_research_and_no_gpu_before_admission():
    h=helper()
    assert h.stage_order()==['admission','qualification','calibration','acceptance']
    order=h.stage_order(research=True)
    assert order.index('admission')<order.index('baseline')<order.index('train-B')
    assert not any('D' in n for n in order)
    assert 'train-D' in h.stage_order(research=True,arms=['B','C','D'])


def test_wrapper_delegates_to_canonical_cli_with_explicit_engine_and_qualified_source():
    h=helper();calls=[]
    result=h.invoke('podman','evaluate','study.json','output',qualification='producer',extra=['--checkpoints','A=descriptor.json'],
        seconds=90000,runner=lambda argv,**kw:calls.append((argv,kw)) or {'status':'PASS'})
    assert result['status']=='PASS'
    argv,options=calls[0]
    assert argv[1:6]==['scripts/foundation_run.py','--engine','podman','generalization','evaluate']
    assert options['seconds']==90000
    with pytest.raises(ValueError):h.canonical_command('auto','train','study','out')
    with pytest.raises(ValueError):h.canonical_command('podman','train','study','out')
