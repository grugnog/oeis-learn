"""Retain bounded campaign acceptance attempts for the round020 successor."""
import fcntl

from oeis_learn.generalization.progressive.contracts import load, reference
from .identity import confined, digest, seal
from .preparation_recovery import Allowance, UnsafeRecovery
from .progressive_workstation_state import Blocked, read
from .research_limits import CLEANUP_SECONDS


def accepted_folder(root, base, journal):
    row = journal['steps'].get('prepare-research/acceptance', {})
    value = row.get('value')
    canonical = base / 'acceptance'
    if not isinstance(value, dict) or 'output' not in value:
        return canonical
    path = confined(root, value['output'])
    if path != canonical and not (path.parent == canonical / 'attempts'
                                  and len(path.name) == 4 and path.name.isascii() and path.name.isdigit()):
        raise ValueError('acceptance producer outside its owned attempt directory')
    return path


def stopped(root, target, logs):
    """Verify the supervisor and each started container before spending another attempt."""
    process = load(confined(root, logs / 'process.json'))
    if type(process.get('returncode')) is not int or process.get('cleanup_failure'):
        raise UnsafeRecovery('acceptance supervisor termination unproven')
    for stage in ('software', 'device', 'runtime-inspection/device/logs',
                  'runtime-inspection/image-0/logs', 'runtime-inspection/image-1/logs',
                  'runtime-inspection/selected-device'):
        directory = confined(root, target / stage, exists=False)
        if not (directory / 'command.json').exists():
            if process.get('outcome') == 'COMPLETE' and process['returncode'] == 0:
                raise UnsafeRecovery('completed acceptance container evidence missing: ' + stage)
            continue
        child = load(confined(root, directory / 'process.json'))
        cleanup_dir = directory.parent if directory.parent.name in ('image-0', 'image-1') else directory
        cleanup = load(confined(root, cleanup_dir / 'cleanup.json'))
        if (type(child.get('returncode')) is not int or child.get('cleanup_failure')
                or cleanup.get('acknowledged') is not True or cleanup.get('within_bound') is not True
                or cleanup.get('failure')):
            raise UnsafeRecovery('acceptance container termination unproven: ' + stage)
    return process


def published(root, target, logs, campaign, scope, scope_model):
    from .progressive_device_acceptance import consume_receipt
    process = stopped(root, target, logs)
    if process.get('returncode') != 0 or process.get('outcome') != 'COMPLETE':
        return None
    result = read(confined(root, target / 'result.json'))
    receipt = consume_receipt(target, read(confined(root, target / 'acceptance.json')))
    request = read(confined(target, receipt['request']['path']))
    if (any(request.get(key) != expected for key, expected in scope.items())
            or digest({k:v for k,v in request['profile']['model'].items() if k != 'seed'}) !=
               scope_model
            or receipt['campaign'] != campaign or result['campaign'] != campaign
            or result['mechanical_status'] != 'COMPLETE' or result['operation'] != 'acceptance'
            or result['metrics']['acceptance'] != reference(target, 'acceptance.json')):
        raise ValueError('original campaign acceptance publication differs')
    files = [logs / 'process.json', target / 'result.json', target / 'acceptance.json',
             *(confined(target, receipt[k]['path']) for k in ('software', 'device', 'request'))]
    return dict(operation='acceptance', output=str(target.relative_to(root)), exit_code=0), files


def run(w, seconds):
    from .progressive_workstation import command
    from .supervision import launch

    root, base, session = w.root, w.base, w.s
    key = 'prepare-research/freeze'
    if session.journal['steps'].get(key, {}).get('status') != 'COMPLETE':
        return session.step('acceptance', lambda: None, dependencies=(key,))
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES
    from .progressive_handoff import named_plan
    campaign_path = base / 'frozen/campaign.json'
    frozen = load(campaign_path, maximum=CAMPAIGN_DOCUMENT_BYTES)
    plan = named_plan(root, frozen, 'acceptance_plan', 'progressive-acceptance-plan/v1',
                      {'qualification', 'seconds'})
    if plan is None:
        raise Blocked('frozen campaign acceptance plan missing')
    seconds = min(seconds, plan['seconds'])
    binding = seal(dict(scope=session.binding, campaign=reference(root, campaign_path),
                        seconds=seconds, operation='acceptance'))
    row = session.journal['steps'].get('prepare-research/acceptance', {})
    if row.get('binding', binding) != binding:
        raise Blocked('acceptance scope changed')
    def action():
        path = base / 'acceptance-allowance.json'
        parent = confined(root, base / 'acceptance/attempts', exists=False)
        runtime = read(base / 'frozen/runtime.json')
        matrix = read(base / 'matrix/matrix.json')
        if runtime['identity'] != frozen['payload']['runtime']:
            raise Blocked('frozen runtime differs')
        scope = dict(runtime=runtime['identity'], source=session.binding['source'], engine=session.binding['engine'],
                     image=runtime['payload']['scope']['images'][1], hardware=matrix['hardware'],
                     execution_policy=runtime['payload']['scope']['execution_policy'])
        campaign = frozen['identity']
        previous = sorted(parent.iterdir()) if parent.exists() else []
        if previous:
            if not path.is_file():
                raise Blocked('original acceptance allowance missing')
            ledger = read(path)
            if ledger['binding'] != binding or ledger['seconds'] != seconds:
                raise Blocked('original acceptance allowance differs')
            issued = [*ledger['attempts'], *([ledger['active']] if ledger['active'] else [])]
            if len(issued) != len(previous):
                raise Blocked('acceptance attempt inventory differs')
            for ordinal, (target, entry) in enumerate(zip(previous, issued)):
                if target.name != f'{ordinal:04d}' or entry['ordinal'] != ordinal:
                    raise Blocked('acceptance attempt order differs')
                original = published(root, target, base / 'logs/acceptance' / target.name, campaign, scope, runtime['payload']['scope']['model'])
                if original:
                    if target != previous[-1]:
                        raise Blocked('additional attempt after completed acceptance')
                    # A hard stop after successful publication can leave the
                    # interval open. Close at its reserved ceiling before launch.
                    if ledger['active'] is not None:
                        from .progressive_workstation_state import write
                        with path.with_suffix('.lock').open('a') as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            if read(path) != ledger:
                                raise Blocked('acceptance ledger changed during recovery')
                            active = ledger['active']
                            ledger['attempts'].append(dict(active, outcome='RECOVERED_PUBLICATION',
                                                          charged_seconds=active['reserved_seconds']))
                            ledger.update(active=None, spent_seconds=seconds)
                            write(path, seal(ledger))
                    return original
            if ledger['active'] is None and seconds - ledger['spent_seconds'] <= CLEANUP_SECONDS:
                raise Blocked('insufficient remaining acceptance allowance for cleanup')
        elif path.exists():
            raise Blocked('original acceptance attempt directory missing')
        try:
            with Allowance(path, binding, seconds).attempt() as (left, ordinal):
                target = parent / f'{ordinal:04d}'
                target.mkdir(parents=True, exist_ok=False)
                logs = base / 'logs/acceptance' / target.name
                args = command(root, w.args, 'acceptance', campaign=str(campaign_path.relative_to(root)),
                               output=str(target.relative_to(root)))
                try:
                    launch(args, logs, seconds=left-CLEANUP_SECONDS, stream_cap=4<<20,
                           cleanup_seconds=CLEANUP_SECONDS, grace_seconds=40, allowed_returncodes=(0, 1, 2, 3))
                except KeyboardInterrupt:
                    try:
                        stopped(root, target, logs)
                    except (OSError, ValueError) as exc:
                        raise UnsafeRecovery('interrupted acceptance cleanup unproven') from exc
                    # Known wall interval with acknowledged cleanup. Exiting
                    # through this exception charges the measured attempt.
                    raise CleanInterrupt from None
                original = published(root, target, logs, campaign, scope, runtime['payload']['scope']['model'])
                if original is None:
                    raise Blocked('acceptance failed; original logs: ' + str(logs))
                return original
        except CleanInterrupt:
            raise KeyboardInterrupt from None

    return session.step('acceptance', action, dependencies=('prepare-research/freeze',),
                        binding=binding, recover=row.get('status') != 'COMPLETE')


class CleanInterrupt(Exception):
    """The original process and every issued container acknowledged termination."""
