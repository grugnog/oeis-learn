"""Safe archive I/O followed by independent Phase B evidence reconciliation."""
import json
from pathlib import Path
import tempfile
import zipfile
from .bundle import verify
from .identity import confined


def verify_archive(archive,*,cap=512<<20):
    from oeis_learn.generalization.bundles import verify_bundle
    archive=Path(archive)
    if not 0<cap<=512<<20 or archive.stat().st_size>cap:raise ValueError('archive byte cap')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>10000 or sum(i.file_size for i in infos)>cap:
            raise ValueError('archive expansion/member cap')
        if 'manifest.json' not in z.namelist() or z.getinfo('manifest.json').file_size>1<<20:
            raise ValueError('archive manifest cap')
    manifest=verify(archive)
    if 'evidence.json' not in {r['path'] for r in manifest['members']}:
        raise ValueError('generalization evidence.json required; recovery archive is not qualification')
    with tempfile.TemporaryDirectory(prefix='generalization-verification-') as temporary:
        root=Path(temporary)
        with zipfile.ZipFile(archive) as z:
            for member in manifest['members']:
                target=confined(root,member['path'],exists=False)
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member['path']) as source,target.open('xb') as destination:
                    import shutil
                    shutil.copyfileobj(source,destination,length=1<<20)
        record=json.loads((root/'evidence.json').read_text())
        if record.get('version')=='generalization-archive/v1':
            from .identity import validate
            from oeis_learn.generalization.contracts import verify_artifact
            validate(record,{'version','evidence','payload_root'},'generalization-archive/v1')
            if record['payload_root']!='payloads' or record['evidence']['path']!='payloads/evidence.json':
                raise ValueError('noncanonical evidence payload root')
            record=json.loads(verify_artifact(root,record['evidence']))
            root=confined(root,'payloads')
        study=json.loads(confined(root,record['study']['path']).read_text())
        result=verify_bundle(record,expected_study=study['identity'],root=root)
        if result['software_status']!=manifest['summary_state']:
            raise ValueError('archive status differs from reconciled generalization evidence')
        return result
