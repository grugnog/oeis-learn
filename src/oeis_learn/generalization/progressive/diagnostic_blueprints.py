"""Outcome-blind diagnostic templates and finite pre-freeze reservation checks.

READY means ready for the existing campaign freeze, never launch authority.
"""
import json
from importlib.resources import files
from .contracts import seal, verify_refs

RESOLVED = {'selected', 'data', 'policy', 'design', 'task_slots', 'panel_slots', 'caps', 'ancillary_caps'}
CAPS = {'tokens','draws','rejections','attempts','native_work','elapsed_us','disk_bytes','archive_bytes'}


def inventory():
    return json.loads(files('oeis_learn.generalization').joinpath('policies/progressive-diagnostics.json').read_text())


def _caps(value):
    if not isinstance(value, dict) or set(value) != CAPS or any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('closed nonnegative resource reservation required')
    return value


def resolve(root, book, *, observed, visited, pool, exposed_panels=()):
    canonical = inventory()
    if set(book) != set(canonical) or any(book[k] != canonical[k] for k in canonical if k != 'branches'):
        raise ValueError('non-executable registered diagnostic inventory required')
    if len(book['branches']) != len(canonical['branches']):
        raise ValueError('complete contingency roster required')
    _caps(pool)
    if pool['elapsed_us'] > 24*3600*1000000:
        raise ValueError('shared pool cannot exceed 24 hours')
    rows, used, tasks, panels, classes = [], dict.fromkeys(CAPS, 0), set(), set(), set()
    for branch, template in zip(book['branches'], canonical['branches']):
        if set(branch) != set(template) or any(branch[k] != template[k] for k in template if k not in RESOLVED):
            raise ValueError('registered question, trigger, observations and terminal rule required')
        if type(branch['selected']) is not bool:
            raise ValueError('explicit branch selection required')
        status = 'DISABLED'
        if branch['id'] == 'integrity-quarantine' and branch['trigger'] in observed:
            status = 'STOP_AFFECTED'
        elif branch['selected']:
            if any(branch[k] is None for k in RESOLVED - {'selected'}):
                status = 'UNRESOLVED'
            else:
                for key in ('task_slots', 'panel_slots'):
                    slots = branch[key]
                    if not isinstance(slots, list) or not slots or any(type(s) is not str or not s for s in slots) or len(set(slots)) != len(slots):
                        raise ValueError('unique concrete task/panel slots required')
                if tasks.intersection(branch['task_slots']) or (panels | set(exposed_panels)).intersection(branch['panel_slots']) or branch['failure_class'] in classes:
                    raise ValueError('diagnostic slot/panel/failure-class reuse forbidden')
                tasks.update(branch['task_slots']); panels.update(branch['panel_slots']); classes.add(branch['failure_class'])
                if len(tasks) > 4:
                    raise ValueError('at most four tasks in the shared pool')
                caps, ancillary = _caps(branch['caps']), _caps(branch['ancillary_caps'])
                if caps['elapsed_us'] > 12*3600*1000000:
                    raise ValueError('diagnostic task cannot exceed 12 hours')
                for key in CAPS:
                    used[key] += len(branch['task_slots']) * caps[key] + ancillary[key]
                    if used[key] > pool[key]:
                        raise ValueError('selected diagnostics exceed existing shared reservation: '+key)
                if not isinstance(branch['data'], list) or not branch['data']:
                    raise ValueError('immutable data required')
                verify_refs(root, branch['data'])
                verify_refs(root, [branch['policy']])
                verify_refs(root, [branch['design']])
                status = ('EXHAUSTED' if branch['failure_class'] in visited or len(visited) >= 4
                          else 'UNTRIGGERED' if branch['trigger'] not in observed else 'READY')
        rows.append(dict(id=branch['id'], status=status, terminal_action=branch['terminal_action']))
    return seal(dict(version='progressive-diagnostic-preflight/v1', executable=False,
                     ready_for_freeze=any(r['status']=='READY' for r in rows) and
                     not any(r['status'] in {'UNRESOLVED','EXHAUSTED','STOP_AFFECTED'} for r in rows),
                     branches=rows, reserved=used, pool=pool,
                     authority='proposal only; original verdict and campaign freeze remain binding'))
