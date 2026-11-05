import math
import pytest
from oeis_learn.pipeline_validation.identity import digest,seal


def proof():
    from oeis_learn.pipeline_validation.generalization_execution import POLICY_ID
    return seal(dict(version='pac-crash-proof/v1',status='PASS',device='hip',diagnostic_only=True,
        source=digest('source'),execution_policy=POLICY_ID,cadence='pac-acceptance/v1',scale='production',
        crash_exit=86,actual_optimizer_calls=19,saved_updates=0,newer_valid_rewards=7,newer_graded_anchors=2,
        restored_old_weights=True,restored_old_policy=True,crashed_seconds=20.,discarded_tokens=900,
        charged_tokens=1652,useful_tokens=752,cap=4096,updates=16,learner_seconds=35.,segment='recovery-1',
        checkpoint=digest('checkpoint'),device_rng_present=True))


def test_actual_hip_crash_proof_is_required_and_closed():
    from oeis_learn.pipeline_validation.generalization_crash_proof import verify
    value=proof();assert verify(value,source=digest('source'))==value
    for changed in [dict(device='cpu'),dict(source=digest('other')),dict(device_rng_present=False),
        dict(restored_old_policy=False),dict(restored_old_weights=False),dict(crash_exit=0),
        dict(newer_valid_rewards=0),dict(newer_graded_anchors=0),dict(discarded_tokens=0),
        dict(charged_tokens=100),dict(learner_seconds=1.),dict(crashed_seconds=math.nan),
        dict(extra='unbound'),dict(scale='tiny')]:
        with pytest.raises(ValueError):verify(seal(dict(value,**changed)),source=digest('source'))
    with pytest.raises(ValueError):verify(dict(value,updates=17),source=digest('source'))


def test_focused_suite_is_explicit_in_portable_parser():
    from tests.unit.test_generalization_host import host_module
    args=host_module().parser().parse_args(['--engine','podman','validate','--suite','pac-crash-hip','--output','run-state/t072'])
    assert args.suite=='pac-crash-hip' and args.engine=='podman'
