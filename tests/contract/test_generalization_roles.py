"""Reviewable payload/mount contracts and disabled routing, not sandbox qualification."""
import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from oeis_learn.generalization.interfaces import (
    ArtifactRef, Command, CommandRequest, EXIT_CODES, FullStateCheckpoint,
    ScientificStatus, SoftwareStatus, WeightsOnlyCheckpoint,
)
from oeis_learn.generalization.role_contracts import ROLE_INPUTS, ROLE_OUTPUTS, Role
from oeis_learn.pipeline_validation.generalization_cli import HANDLERS, dispatch
from test_generalization_authority import validator

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/generalization"


def test_visible_prompt_has_no_truth_or_protected_labels():
    record = json.loads((FIXTURES / "payloads.json").read_text())["visible-prompt"]
    validator("visible-prompt").validate(record)
    for field in ["source", "family", "stratum", "hidden_values", "target_program", "assignment"]:
        changed = copy.deepcopy(record)
        changed[field] = "forbidden"
        assert list(validator("visible-prompt").iter_errors(changed))
    for value in ["01", "-0", "+1", "1\n", 1, True]:
        changed = copy.deepcopy(record)
        changed["values"][0] = value
        assert list(validator("visible-prompt").iter_errors(changed))


def test_training_and_scorer_payloads_are_not_generator_inputs():
    records = json.loads((FIXTURES / "payloads.json").read_text())
    for name in ["training-target", "scorer-target"]:
        validator(name).validate(records[name])
        assert list(validator("visible-prompt").iter_errors(records[name]))
    changed = copy.deepcopy(records["training-target"])
    changed["role"] = "assessment"
    assert list(validator("training-target").iter_errors(changed))


def test_mounts_match_independently_declared_permissions():
    expected = json.loads((FIXTURES / "role-expectations.json").read_text())
    assert {r.value: [m.channel for m in mounts] for r, mounts in ROLE_INPUTS.items()} == expected
    assert all(m.read_only for mounts in ROLE_INPUTS.values() for m in mounts)
    assert all(not m.read_only for m in ROLE_OUTPUTS.values())
    assert len({m.destination for m in ROLE_OUTPUTS.values()}) == len(Role)
    with pytest.raises(TypeError):
        ROLE_INPUTS[Role.LEARNER] = ()


def test_immutable_types_and_distinct_checkpoint_kinds():
    ref = ArtifactRef("weights.bin", "sha256:" + "0" * 64, 1)
    weights = WeightsOnlyCheckpoint(ref, "random")
    full = FullStateCheckpoint(ref, ref, ref.sha256, 0, 0, ("latest",))
    assert not isinstance(weights, FullStateCheckpoint)
    for record, field in [(ref, "path"), (full, "tokens"), (weights, "initialization")]:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)


@pytest.mark.parametrize("operation", list(Command))
def test_all_prospective_commands_block_without_io(operation, tmp_path):
    output = tmp_path / "must-not-exist"
    request = CommandRequest(operation, str(output))
    result = dispatch(request)
    assert result.software_status == SoftwareStatus.BLOCKED
    assert result.scientific_status == ScientificStatus.UNMEASURED
    assert result.reason == "generalization_phase_c_not_implemented"
    assert result.artifacts == ()
    assert not output.exists()
    assert set(HANDLERS) == set(Command)
    assert EXIT_CODES[result.software_status] == 3


def test_unknown_command_rejected_and_exit_codes_preserved():
    with pytest.raises(ValueError):
        dispatch(CommandRequest("skip-tests", "unused"))
    assert dict(EXIT_CODES) == {
        "PASS": 0, "FAIL": 1, "BLOCKED": 3, "PARTIAL": 3,
        "INCONCLUSIVE": 3, "INTERRUPTED": 130,
    }


def test_source_shapes_allow_existing_profile_size_without_weakening_prompts():
    records = json.loads((FIXTURES / "payloads.json").read_text())
    record = copy.deepcopy(records["training-target"])
    record["source"] = "(program " + " " * 1024 + "(return (int 0)))"
    validator("training-target").validate(record)
    record["source"] = "x" * 65537
    assert list(validator("training-target").iter_errors(record))


@pytest.mark.parametrize('engine_name',['docker','podman'])
@pytest.mark.parametrize('role_name',['preparation','learner','generator','report'])
def test_real_role_command_uses_qualified_private_runtime(tmp_path,monkeypatch,engine_name,role_name):
    import stat
    from types import SimpleNamespace
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import generalization_roles as roles,supervision
    from oeis_learn.generalization.role_contracts import ROLE_INPUTS,Role
    engine=Engine(engine_name,executable=engine_name,details={})
    monkeypatch.setattr(engine,'user_args',lambda:[])
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    selected=Role(role_name);channels={m.channel:m.channel for m in ROLE_INPUTS[selected]}
    for name in channels.values():(tmp_path/name).mkdir()
    observed=[]
    def launch(argv,output,**kw):
        if selected==Role.GENERATOR:
            runtime=json.loads(argv[-1])
            assert 0 < runtime['seconds'] < kw['seconds']-4
        observed.extend(argv);(output.parent/'result.json').write_text('{"status":"PASS"}')
    roles.invoke(SimpleNamespace(engine=engine,policy={'memory_bytes':2<<30,'cpus':4}),
        'sha256:'+'a'*64,tmp_path,selected,channels,output='out',operation='evaluate' if selected==Role.GENERATOR else 'fixture',settings={},seconds=30,launcher=launch)
    env=dict(observed[i+1].split('=',1) for i,x in enumerate(observed) if x=='--env')
    assert env['TMPDIR']==f'/output/{role_name}/.runtime/tmp'
    assert env['HOME']==f'/output/{role_name}/.runtime/home'
    assert env['XDG_CACHE_HOME']==f'/output/{role_name}/.runtime/cache'
    assert env['OMP_NUM_THREADS']=='2' and env['RAYON_NUM_THREADS']=='1'
    assert '/tmp:rw,noexec,nosuid,nodev,size=2147483648,mode=1777' in observed
    mounts=[observed[i+1] for i,x in enumerate(observed) if x=='--mount']
    assert len(mounts)==len(channels)+1 and all(m.endswith(',readonly') for m in mounts[:-1])
    assert '--read-only' in observed and '--network=none' in observed
    for path in (tmp_path/'out/.runtime').iterdir():assert stat.S_IMODE(path.stat().st_mode)==0o700


@pytest.mark.parametrize('engine_name',['docker','podman'])
@pytest.mark.parametrize('stage',['preparation','learner','generation','acceptance','calibration','resume','proof','export'])
def test_pac_actual_role_limits_are_unclipped(tmp_path,monkeypatch,engine_name,stage):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import generalization_roles as roles,supervision
    from oeis_learn.pipeline_validation.generalization_execution import startup_environment
    from tests.integration.test_generalization_recipes import pac_envelope
    engine=Engine(engine_name,executable=engine_name,details={});monkeypatch.setattr(engine,'user_args',lambda:[])
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    monkeypatch.setattr(engine,'device_args',lambda render_node:['--device','/dev/kfd','--device','/dev/dri/renderD128'])
    selected=Role.LEARNER;channels={m.channel:m.channel for m in ROLE_INPUTS[selected]}
    for name in channels.values():(tmp_path/name).mkdir()
    envelope=pac_envelope()[stage];seen=[]
    def launch(argv,output,**kw):
        seen.extend(argv);runtime=json.loads(argv[-1])
        assert runtime['seconds']==envelope['work']+envelope['publication']
        assert kw['seconds']>=runtime['seconds']
        assert kw['cleanup_seconds']==30
        (output.parent/'result.json').write_text('{"status":"PASS"}')
    host=SimpleNamespace(engine=engine,policy={'memory_bytes':2<<30,'cpus':4},gpu_lock=nullcontext)
    roles.invoke(host,'sha256:'+'a'*64,tmp_path,selected,channels,output='out',operation='train',settings={},
                 seconds=envelope['outer'],gpu=True,launcher=launch,envelope=envelope)
    env=dict(seen[i+1].split('=',1) for i,x in enumerate(seen) if x=='--env')
    assert all(env[k]==v for k,v in startup_environment().items())
    assert env['TMPDIR']=='/output/learner/.runtime/tmp'


def test_pac_process_failure_cannot_be_overridden_by_stale_application_pass(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import generalization_roles as roles,supervision
    from tests.integration.test_generalization_recipes import pac_envelope
    engine=Engine('podman',executable='podman',details={});monkeypatch.setattr(engine,'user_args',lambda:[])
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    channels={m.channel:m.channel for m in ROLE_INPUTS[Role.REPORT]}
    for name in channels.values():(tmp_path/name).mkdir()
    def launch(argv,output,**kw):
        output.mkdir();(output.parent/'result.json').write_text('{"status":"PASS"}')
        (output/'process.json').write_text(json.dumps(dict(outcome='FAIL',returncode=1,cleanup_failure=None)))
    with pytest.raises(ValueError,match='process'):
        roles.invoke(SimpleNamespace(engine=engine,policy={'memory_bytes':2<<30,'cpus':4}),
            'sha256:'+'a'*64,tmp_path,Role.REPORT,channels,output='out',operation='fixture',settings={},
            seconds=3900,launcher=launch,envelope=pac_envelope()['export'])


@pytest.mark.parametrize('engine_name',['docker','podman'])
@pytest.mark.parametrize('role_name,operation,stage,gpu',[
    ('preparation','pac-assemble','preparation',False),('learner','pac-calibrate','calibration',True),
    ('learner','pac-campaign-acceptance','acceptance',True),('preparation','pac-publish-evaluation','validation',False),
    ('generator','pac-evaluate','generation',True),('selection','pac-score','proof',False)])
def test_pac_delivery_commands_have_exact_mounts_and_full_envelope(tmp_path,monkeypatch,engine_name,role_name,operation,stage,gpu):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import generalization_roles as roles,supervision
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.generalization_execution import startup_environment
    engine=Engine(engine_name,executable=engine_name,details={})
    monkeypatch.setattr(engine,'user_args',lambda:[])
    monkeypatch.setattr(engine,'device_args',lambda render_node:['--device','/dev/kfd','--device','/dev/dri/renderD128'])
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    role=Role(role_name);channels={m.channel:m.channel for m in ROLE_INPUTS[role]}
    for name in channels.values():(tmp_path/name).mkdir()
    limits=envelope(purpose='research',arms=['B','C'])[stage];observed=[]
    def launch(argv,output,**kw):
        observed.extend(argv);runtime=json.loads(argv[-1])
        assert runtime['operation']==operation and runtime['seconds']==limits['work']+limits['publication']
        assert runtime['device']==('cuda' if gpu else 'cpu')
        assert kw['seconds']>=runtime['seconds'] and kw['cleanup_seconds']==30
        (output.parent/'result.json').write_text('{"status":"PASS"}')
    host=SimpleNamespace(engine=engine,policy=dict(memory_bytes=8<<30,cpus=4),gpu_lock=nullcontext)
    roles.invoke(host,'sha256:'+'a'*64,tmp_path,role,channels,output='out',operation=operation,settings={},
        seconds=limits['outer'],envelope=limits,gpu=gpu,launcher=launch)
    mounts=[observed[i+1] for i,x in enumerate(observed) if x=='--mount']
    assert len(mounts)==len(channels)+1 and all(m.endswith(',readonly') for m in mounts[:-1])
    assert '--network=none' in observed and '--read-only' in observed and not any('sock' in m for m in mounts)
    if gpu:
        env=dict(observed[i+1].split('=',1) for i,x in enumerate(observed) if x=='--env')
        assert all(env[k]==v for k,v in startup_environment().items())
