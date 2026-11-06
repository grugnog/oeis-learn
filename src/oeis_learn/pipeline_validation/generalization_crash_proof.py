"""Closed bounded diagnostic evidence; never upgrades a full research authority."""
import math
import re
from .identity import seal
from .generalization_execution import POLICY_ID

FIELDS=set('version identity status device diagnostic_only source execution_policy cadence scale crash_exit actual_optimizer_calls saved_updates newer_valid_rewards newer_graded_anchors restored_old_weights restored_old_policy crashed_seconds discarded_tokens charged_tokens useful_tokens cap updates learner_seconds segment checkpoint device_rng_present'.split())


def verify(record,*,source,device='hip'):
    if not isinstance(record,dict) or set(record)!=FIELDS or seal(record)!=record:raise ValueError('intact closed crash proof required')
    if (record['version'],record['status'],record['device'],record['source'])!=('pac-crash-proof/v1','PASS',device,source):raise ValueError('crash proof producer/device mismatch')
    if device not in {'hip','cpu'}:raise ValueError('explicit proof device required')
    if any(not isinstance(record[k],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',record[k]) for k in ('source','checkpoint')):raise ValueError('crash proof digest required')
    if record['diagnostic_only'] is not True or record['restored_old_weights'] is not True or record['restored_old_policy'] is not True:raise ValueError('actual old model and policy restoration required')
    if record['cadence']!='pac-acceptance/v1' or record['segment']!='recovery-1':raise ValueError('declared reduced recovery scope required')
    if device=='hip' and (record['scale']!='production' or record['execution_policy']!=POLICY_ID or record['device_rng_present'] is not True):raise ValueError('actual HIP policy/RNG evidence required')
    if device=='cpu' and (record['scale']!='tiny' or record['execution_policy'] is not None or record['device_rng_present'] is not False):raise ValueError('CPU wiring must remain labeled')
    for k in ('crash_exit','actual_optimizer_calls','saved_updates','newer_valid_rewards','newer_graded_anchors','discarded_tokens','charged_tokens','useful_tokens','cap','updates'):
        if type(record[k]) is not int or not 0<=record[k]<=10**9:raise ValueError('bounded exact crash counter required')
    if (record['crash_exit'],record['actual_optimizer_calls'],record['saved_updates'],record['updates'])!=(86,19,0,16):raise ValueError('actual abrupt-exit and recovery boundaries required')
    if not record['newer_valid_rewards'] or not record['newer_graded_anchors'] or not record['discarded_tokens'] or not record['useful_tokens']:raise ValueError('actual discarded newer work required')
    if record['charged_tokens']!=record['useful_tokens']+record['discarded_tokens'] or record['charged_tokens']>record['cap']:raise ValueError('lost work must remain charged within cap')
    for k in ('crashed_seconds','learner_seconds'):
        if type(record[k]) not in (int,float) or not math.isfinite(record[k]) or not 0<record[k]<=7200:raise ValueError('finite actual recovery time required')
    if record['learner_seconds']<record['crashed_seconds']:raise ValueError('crash time was refunded')
    return record
