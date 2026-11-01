"""Attach final controller evidence without changing the verified report payloads."""
import json
import shutil
import zipfile
from pathlib import Path
from .identity import seal,write,confined
from .generalization_preparation import reference
from .bundle import pack,verify
from oeis_learn.generalization.recovery import verified_scope,verify_final_workflow,verify_verifier_terminal

FINAL_FILES=('workflow-final.json','verification-final.json','report-manifest.json','verification-terminal.json')


def verification(root,exit_code):
    root=Path(root);path=root/'verify-bundle/summary.json'
    if path.is_symlink() or path.stat().st_size>65536:raise ValueError('bounded verifier summary required')
    summary=json.loads(path.read_text())
    terminal=root/'verify-bundle/result.json'
    if terminal.is_symlink() or terminal.stat().st_size>65536:raise ValueError('bounded verifier terminal record required')
    verify_verifier_terminal(json.loads(terminal.read_text()),summary=reference(path,path.parent),archive=summary['archive'],exit_code=exit_code)
    manifest=verify(root/'report.zip')
    verified_scope(summary,exit_code,archive=reference(root/'report.zip',root),scope=manifest['summary_state'])
    return summary


def validate_final(root,manifest):
    names={r['path']:r for r in manifest['members']}
    if not set(FINAL_FILES)&set(names):return
    if not set(FINAL_FILES)<=set(names):raise ValueError('incomplete final workflow evidence')
    workflow,summary,original,terminal=[json.loads((root/name).read_text()) for name in FINAL_FILES]
    summary_ref=dict(reference(root/'verification-final.json',root),path='summary.json')
    verify_verifier_terminal(terminal,summary=summary_ref,archive=summary['archive'],exit_code=workflow['receipts'][-1]['exit_code'])
    if seal(original)!=original or original.get('version')!='pipeline-bundle/v1':raise ValueError('original report manifest seal')
    verified_scope(summary,workflow['receipts'][-1]['exit_code'],archive=summary['archive'],scope=original['summary_state'])
    verify_final_workflow(workflow,summary,manifest['summary_state'])
    for row in original['members']:
        if row['path'] not in names or any(row[k]!=names[row['path']][k] for k in ('sha256','bytes')):
            raise ValueError('finalization changed original report payload')


def publish(root,workflow,summary):
    root=Path(root);archive=root/'report.zip';manifest=verify(archive)
    if verification(root,workflow['receipts'][-1]['exit_code'])!=summary:raise ValueError('verifier summary changed')
    verified_scope(summary,workflow['receipts'][-1]['exit_code'],archive=reference(archive,root),scope=manifest['summary_state'])
    verify_final_workflow(workflow,summary,manifest['summary_state'])
    if set(FINAL_FILES)&{r['path'] for r in manifest['members']}:raise ValueError('report already finalized')
    target=root/'final-export';target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for member in manifest['members']:
            path=confined(target,member['path'],exists=False);path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member['path']) as source,path.open('xb') as destination:
                shutil.copyfileobj(source,destination,1<<20)
    write(target/'workflow-final.json',workflow);write(target/'report-manifest.json',manifest)
    shutil.copyfile(root/'verify-bundle/summary.json',target/'verification-final.json')
    shutil.copyfile(root/'verify-bundle/result.json',target/'verification-terminal.json')
    # Retain all original members, including those previously optional. No receipt
    # or measurement is rewritten to reflect controller packaging operations.
    result=pack(target,mandatory=[r['path'] for r in manifest['members']]+list(FINAL_FILES))
    final=verify(result);validate_final(target,final)
    destination=root.with_suffix('.zip')
    if destination.exists():raise ValueError('final archive already exists')
    result.rename(destination)
    return destination
