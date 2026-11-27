"""Retire completed, superseded qualification tensors; retain all run history.

Stdlib CLI may be copied into run-state and executed against a qualified checkout.
Only weights.pt/optimizer.pt in original successful HIP qualification cases are
eligible. No whole run, calibration checkpoint, scientific model or input is deleted.
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Direct execution is intentional: the active checkout remains source-qualified.
if __name__ == '__main__':
    sys.path.insert(0, str(Path.cwd()/'src'))

from oeis_learn.pipeline_validation import discovery_cleanup as cleanup
from oeis_learn.pipeline_validation.progressive_workstation_state import read, write
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file, _json
from oeis_learn.pipeline_validation.identity import seal, digest
from oeis_learn.generalization.progressive.contracts import reference, CAMPAIGN_DOCUMENT_BYTES

VERSION = 'discovery-qualification-retirement/v1'


def protected(root, active):
    """Keep the current session and its declared data/qualification ancestors."""
    folder = cleanup.safe(root, active)
    binding = read(folder/'session.json')['binding']
    journal = read(folder/'journal.json')
    if any(r.get('status') in {'RUNNING', 'INTERRUPTED'} for r in journal['steps'].values()):
        raise ValueError('active campaign must be stopped cleanly before retirement')
    path = folder/'prepare-research/prepared/prepared.json'
    prepared = _json(path, maximum=CAMPAIGN_DOCUMENT_BYTES)
    if seal(prepared) != prepared:
        raise ValueError('original sealed prepared campaign required')
    keep = {active}
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():visit(child)
        elif isinstance(value, list):
            for child in value:visit(child)
        elif isinstance(value, str) and value.startswith('run-state/discovery/'):
            parts = value.split('/')
            if len(parts) >= 3:keep.add('/'.join(parts[:3]))
    visit(binding);visit(prepared)
    return sorted(keep), [reference(root, folder/'session.json'), reference(root, folder/'journal.json'), reference(root, path)]


def candidates(root, session):
    from oeis_learn.pipeline_validation.progressive_handoff import original_authority, validate_device_proofs
    from oeis_learn.pipeline_validation.progressive_trajectory import clean_stage
    from oeis_learn.pipeline_validation.generalization_qualification_io import public_closure
    folder = cleanup.safe(root, session)
    if folder.parent != root/'run-state/discovery':
        raise ValueError('one explicit discovery session required')
    journal = read(folder/'journal.json')
    if (any(not k.startswith('prepare-research/') for k in journal['steps'])
            or any(r.get('status') in {'RUNNING', 'INTERRUPTED'} for r in journal['steps'].values())
            or (folder/'run-research').exists()):
        raise ValueError('inactive preparation-only session required: '+session)
    base = folder/'prepare-research';qroot = base/'qualification'
    step = journal['steps'].get('prepare-research/qualify', {})
    result_ref = reference(root, qroot/'result.json')
    if step.get('status') != 'COMPLETE' or result_ref not in step.get('outputs', []):
        raise ValueError('journal-bound complete qualification required: '+session)
    result = read(qroot/'result.json')
    if (result.get('mechanical_status') != 'COMPLETE' or result.get('qualification_status') != 'QUALIFIED'
            or result.get('metrics', {}).get('qualification') != reference(qroot, qroot/'qualification.json')):
        raise ValueError('original successful qualification required')
    qref = reference(root, qroot/'qualification.json')
    q, checked_root = original_authority(root, qref)
    if checked_root != qroot:
        raise ValueError('qualification owner differs')
    source = digest(_json(checked_file(qroot, q['source_closure'])))
    if source != read(folder/'session.json')['binding']['source']:
        raise ValueError('original qualification source binding differs')
    _, producers = validate_device_proofs(qroot, q, source=source, image=q['images'][1], engine=q['engine_security']['engine'])
    clean_stage(qroot/'device')
    evidence = [reference(root, folder/n) for n in ('session.json', 'journal.json')]
    evidence += [result_ref, qref]
    evidence += [reference(root, checked_file(qroot, r)) for r in public_closure(qroot, q)]
    evidence += [reference(root, qroot/'device'/n) for n in ('process.json', 'cleanup.json', 'container-inspect.json')]
    roots = []
    for ref in producers:
        path = checked_file(qroot, ref)
        # Reused external producers are retained, never followed into another run.
        if not path.is_relative_to(qroot/'device/cases'):
            continue
        roots.append(cleanup.safe(root, path.parent/'checkpoints', exists=False))
        evidence.append(reference(root, path))
    files, evidence = cleanup.tensor_inventory(root, roots, evidence)
    if {r['path'] for r in files} & {r['path'] for r in evidence}:
        raise ValueError('tensor is part of immutable qualification evidence')
    return files, evidence


def images(root, active):
    """Observe exact qualified IDs after pruning; never build or pull replacements."""
    from oeis_learn.container_runtime import Engine
    folder = cleanup.safe(root, active)
    binding = read(folder/'session.json')['binding']
    q = read(folder/'prepare-research/qualification/qualification.json')
    engine = Engine(binding['engine']);rows = []
    for image in q['images']:
        try:
            found = engine.image(image)
            rows.append(dict(expected=image, actual=found, status='PRESENT' if found == image else 'DIFFERENT'))
        except Exception as exc:
            rows.append(dict(expected=image, status='MISSING_OR_UNAVAILABLE', error=str(exc)[:4096]))
    return dict(engine=binding['engine'], images=rows, all_present=all(r['status']=='PRESENT' for r in rows))


def enroll(root, sessions, active, output):
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError('distinct explicit sessions required')
    if any(output.is_relative_to(cleanup.safe(root, s)) for s in [active, *sessions]):
        raise ValueError('maintenance output must be outside protected/retired sessions')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise ValueError('plan exists; use apply or a different output')
    with cleanup.locks(root, [active, *sessions]):
        keep, evidence = protected(root, active)
        if set(sessions) & set(keep):
            raise ValueError('selected session is a current campaign dependency: '+', '.join(sorted(set(sessions)&set(keep))))
        files = []
        for session in sessions:
            print('Verifying completed qualification: '+session, flush=True)
            selected, refs = candidates(root, session)
            files += selected;evidence += refs
        if len({r['path'] for r in files}) != len(files):
            raise ValueError('overlapping retirement files')
        value = seal(dict(version=VERSION, root=str(root), sessions=sessions, active_session=active,
            protected_sessions=keep, files=files, evidence=evidence,
            logical_bytes=sum(r['bytes'] for r in files),
            reclaimable_bytes_lower_bound=sum(r['blocks']*512 for r in files if r['links']==1),
            consequence='superseded qualification test models become non-resumable; all metadata, scientific checkpoints, inputs and current campaign dependencies remain'))
        write(output/'plan.json', value)
        # Kept in inventory.json so the existing cleanup ZIP includes it.
        write(output/'inventory.json', dict(qualified_images=images(root, active),
            free_bytes=__import__('shutil').disk_usage(root).free,
            by_session=[dict(session=s, files=sum(r['path'].startswith(s+'/') for r in files),
                             logical_bytes=sum(r['bytes'] for r in files if r['path'].startswith(s+'/'))) for s in sessions]))
        cleanup.bundle(output)
        return value


def apply(root, output):
    value = read(output/'plan.json')
    if value.get('version') != VERSION or value.get('root') != str(root):
        raise ValueError('original qualification retirement plan required')
    with cleanup.locks(root, [value['active_session'], *value['sessions']]):
        keep, _ = protected(root, value['active_session'])
        if set(value['sessions']) & set(keep):
            raise ValueError('retirement session is now a campaign dependency')
        for ref in value['evidence']:checked_file(root, ref)
        allowed = []
        for session in value['sessions']:allowed += candidates(root, session)[0]
        planned = {r['path']: r for r in value['files']}
        remaining = {p for p in planned if cleanup.safe(root, p, exists=False).exists()}
        if {r['path'] for r in allowed} != remaining:
            raise ValueError('qualification tensor inventory changed')
        intent_path = output/'retirement-intent.json'
        if intent_path.exists():
            intent = read(intent_path)
            if intent.get('plan') != value['identity']:
                raise ValueError('original retirement intent differs')
        else:
            refs = []
            for i, row in enumerate(value['files']):
                path = cleanup.safe(root, row['path'])
                if cleanup.stamp(path) != {k:row[k] for k in cleanup.stamp(path)}:
                    raise ValueError('planned tensor changed')
                ref = reference(root, path)
                if ref['sha256'] != row['original_sha256']:
                    raise ValueError('checkpoint tensor differs from original manifest')
                refs.append(ref)
                if i % 10 == 0:print(f'Checked tensor {i+1}/{len(value["files"])} before deletion', flush=True)
            intent = seal(dict(version=VERSION, plan=value['identity'], files=refs, status='INVENTORIED'))
            write(intent_path, intent)
        originals = {r['path']: r for r in intent['files']}
        if set(originals) != set(planned):
            raise ValueError('retirement file set differs')
        # Revalidate remaining bytes on interrupted apply; no partial unchecked set.
        for row in value['files']:
            path = cleanup.safe(root, row['path'], exists=False)
            if path.exists():
                info = cleanup.stamp(path)
                if any(info[k] != row[k] for k in ('bytes','device','inode','mtime_ns')) or reference(root,path) != originals[row['path']]:
                    raise ValueError('retirement tensor changed')
        for row in value['files']:
            path = cleanup.safe(root, row['path'], exists=False)
            if path.exists():
                path.unlink()
                fd = os.open(path.parent, os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
        # Replay the actual qualification evidence after deleting only test models.
        for session in value['sessions']:
            if candidates(root, session)[0]:
                raise ValueError('unexpected remaining qualification tensors')
        receipt = seal(dict(version=VERSION, plan=value['identity'], status='RETIRED', files=len(value['files']),
            logical_bytes=value['logical_bytes'], scientific_history_deleted=False,
            qualification_evidence_revalidated=True, diagnostic_models_resumable=False))
        write(output/'retirement.json', receipt);cleanup.bundle(output)
        return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan','apply'))
    p.add_argument('--session', action='append', default=[])
    p.add_argument('--active-session', default='run-state/discovery/013-research-round-019')
    p.add_argument('--output', required=True)
    args = p.parse_args(argv);root = Path.cwd().resolve()
    output = cleanup.safe(root, args.output, exists=False)
    if not output.is_relative_to(root/'run-state/maintenance'):
        p.error('output must be under run-state/maintenance')
    try:
        result = enroll(root, args.session, args.active_session, output) if args.action=='plan' else apply(root, output)
        print(json.dumps({k:v for k,v in result.items() if k not in {'files','evidence'}}, indent=2))
        if args.action == 'plan':print(json.dumps(read(output/'inventory.json'), indent=2))
        print('Report: '+str(output/'cleanup-report.zip'), flush=True)
        return 0
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write(output/'failure.json', dict(error=type(exc).__name__, message=str(exc)))
        cleanup.bundle(output)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
