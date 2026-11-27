"""Host-owned freeze receipts. Only fully verified, in-memory study work is memoized."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

from oeis_learn.pipeline_validation.identity import confined, digest, seal
from .contracts import atomic_write, load, reference, CAMPAIGN_DOCUMENT_BYTES


def sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename(source, destination):
    source.rename(destination)
    sync_directory(destination.parent)
    if source.parent != destination.parent:
        sync_directory(source.parent)


class FreezeProgress:
    def __init__(self, root, output):
        self.root = root
        self.output = confined(root, output, exists=False)
        self.folder = confined(root, str(self.output) + '.freeze-progress', exists=False)
        self.private = confined(root, self.folder / 'private', exists=False)
        self.stage = self.private / 'publication'
        self.database = None

    @contextmanager
    def lock(self):
        self.private.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = confined(self.root, self.folder / 'freeze.lock', exists=False)
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield self
            finally:
                if self.database is not None:
                    self.database.close()

    def bind(self, args):
        from oeis_learn.experiments.source_snapshot import source_snapshot
        blueprint = load(confined(self.root, args.blueprint))
        prepared = load(confined(self.root, Path(args.prepared) / 'prepared.json'), maximum=CAMPAIGN_DOCUMENT_BYTES)
        qualification = confined(self.root, Path(args.qualification) / 'qualification.json', exists=False)
        qualification_ref = (reference(self.root, qualification) if qualification.exists()
                             else dict(missing=str(qualification.relative_to(self.root))))
        self.binding = seal(dict(version='freeze-checkpoint/v1',
            output=str(self.output.relative_to(self.root)),
            blueprint_identity=blueprint.get('identity'), prepared_identity=prepared.get('identity'),
            blueprint=reference(self.root, args.blueprint),
            prepared=reference(self.root, Path(args.prepared) / 'prepared.json'),
            qualification=qualification_ref,
            source=digest(source_snapshot()), python=sys.version))
        path = confined(self.root, self.folder / 'binding.json', exists=False)
        if path.exists():
            if load(path) != self.binding:
                raise ValueError('freeze checkpoint inputs or validator changed; retain this output and use a fresh scope')
        else:
            if self.output.exists() and any(self.output.iterdir()):
                raise ValueError('existing freeze has no checkpoint binding; original output retained')
            atomic_write(path, self.binding)
            sync_directory(self.folder.parent)
        return blueprint, prepared

    def phase(self, name):
        from oeis_learn.pipeline_validation.progressive_workstation_state import write
        write(self.folder / 'freeze-progress.json', seal(dict(
            version='freeze-progress/v1', binding=self.binding['identity'], phase=name,
            validated_this_attempt=self.validated, reused_this_attempt=self.reused)))

    def mark_complete(self):
        # A process may have committed the publication before updating the
        # human-readable summary. Preserve the original attempt's counters.
        previous = load(confined(self.root, self.folder / 'freeze-progress.json'))
        if previous.get('phase') != 'complete':
            self.validated = previous['validated_this_attempt']
            self.reused = previous['reused_this_attempt']
            self.phase('complete')

    def start(self):
        self.validated = self.reused = 0
        path = self.private / 'validated.sqlite'
        for suffix in ('', '-journal', '-wal', '-shm'):
            confined(self.root, str(path) + suffix, exists=False)
        self.database = sqlite3.connect(path)
        self.database.execute('PRAGMA synchronous=FULL')
        self.database.execute('CREATE TABLE IF NOT EXISTS passed (key TEXT PRIMARY KEY, receipt TEXT NOT NULL)')
        self.database.commit()
        sync_directory(self.private)
        self.phase('validating inputs')

    def study_key(self, plan_id, choice, channel):
        # The prepared identity binds the entire ordered roster and study
        # policy references. The verified channel also binds the package bytes.
        return digest([self.binding['identity'], 'study-variant', plan_id, choice, channel])

    def completed(self, key):
        row = self.database.execute('SELECT receipt FROM passed WHERE key=?', (key,)).fetchone()
        if row is None:
            return False
        if row[0] != digest(['validated', self.binding['identity'], key]):
            raise ValueError('damaged freeze validation receipt')
        self.reused += 1
        return True

    def save(self, key):
        with self.database:
            self.database.execute('INSERT INTO passed VALUES (?,?)',
                                  (key, digest(['validated', self.binding['identity'], key])))
        self.validated += 1
        if self.validated == 1 or self.validated % 64 == 0:
            self.phase('validating study variants')

    def archive(self, path):
        if path.exists():
            rename(path, self.private / ('attempt-' + uuid.uuid4().hex))

    def owned(self, path):
        if load(confined(path, 'freeze-binding.json')) != self.binding:
            raise ValueError('freeze output belongs to a different invocation scope')

    def finished(self, path):
        """Verify an original completed publication before reuse or promotion."""
        confined(self.root, path, exists=False)
        if not (path / 'result.json').exists():
            return None
        result = load(confined(path, 'result.json'))
        if result.get('mechanical_status') != 'COMPLETE':
            return None
        self.owned(path)
        manifest = load(confined(path, 'freeze-publication.json'), maximum=CAMPAIGN_DOCUMENT_BYTES)
        if (seal(result) != result or result.get('operation') != 'freeze'
                or seal(manifest) != manifest or manifest.get('binding') != self.binding['identity']
                or manifest.get('campaign') != result.get('campaign')):
            raise ValueError('freeze publication identity differs')
        refs = manifest['files']
        campaign = load(confined(path, 'campaign.json'), maximum=CAMPAIGN_DOCUMENT_BYTES)
        runtime = load(confined(path, 'runtime.json'))
        if (seal(campaign) != campaign or campaign['identity'] != result['campaign']
                or seal(runtime) != runtime or campaign['payload']['runtime'] != runtime['identity']):
            raise ValueError('freeze campaign/runtime identity differs')
        expected = {'campaign.json','runtime.json'}
        for plan in campaign['payload']['tasks']:
            identity = digest(dict(version='discovery-task/v1', payload=dict(plan, campaign=campaign['identity']),
                                   created_at=campaign['created_at'], parents=[campaign['identity']],
                                   track=campaign['track'], purpose=campaign['purpose']))
            expected.add('tasks/' + identity[7:] + '.json')
        if ({r['path'] for r in refs} != expected or len(refs) != len(expected)
                or {str(p.relative_to(path)) for p in confined(path,'tasks').iterdir()} != expected - {'campaign.json','runtime.json'}):
            raise ValueError('incomplete freeze publication inventory')
        from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
        for ref in refs:
            artifact = checked_file(path, ref)
            if artifact.parent == path / 'tasks':
                task = load(artifact, maximum=CAMPAIGN_DOCUMENT_BYTES)
                if seal(task) != task or artifact.stem != task['identity'][7:]:
                    raise ValueError('published task differs from frozen plan')
        return result

    def begin(self, invocation):
        # Publish the owner and its exact binding together, so a kill between
        # their writes cannot leave an unrecognizable output on the next run.
        if self.output.exists() and any(self.output.iterdir()):
            self.owned(self.output)
            self.archive(self.output)
        working = self.private / ('working-' + uuid.uuid4().hex)
        working.mkdir()
        atomic_write(working / 'freeze-binding.json', self.binding)
        atomic_write(working / 'owner.json', seal(dict(version='discovery-owner/v1',
                     output=self.binding['output'], invocation=invocation)))
        rename(working, self.output)
        self.archive(self.stage)
        self.start()

    def promote(self):
        if self.output.exists():
            self.owned(self.output)
            self.archive(self.output)
        rename(self.stage, self.output)

    def save_validated(self, campaign, runtime, tasks):
        # Task records are deterministic projections of the validated roster.
        # Keep their ordered identities as a check on reconstruction.
        snapshot = seal(dict(version='freeze-validated/v1', binding=self.binding['identity'],
                             campaign=campaign, runtime=runtime, tasks=[t['identity'] for t in tasks]))
        atomic_write(self.private / 'validated-result.json', snapshot)

    def restore_validated(self):
        from .campaign import frozen_task_records
        from .freeze import IncompleteFreeze

        path = confined(self.root, self.private / 'validated-result.json', exists=False)
        if not path.exists():
            return None
        snapshot = load(path, maximum=2 * CAMPAIGN_DOCUMENT_BYTES)
        if seal(snapshot) != snapshot or snapshot['binding'] != self.binding['identity']:
            raise ValueError('validated freeze result identity differs')
        campaign, runtime = snapshot['campaign'], snapshot['runtime']
        if seal(runtime) != runtime or campaign['payload']['runtime'] != runtime['identity']:
            raise ValueError('validated freeze runtime differs')
        storage = campaign['payload']['storage']
        if storage['required_bytes'] > shutil.disk_usage(self.root).free:
            raise IncompleteFreeze(['storage_bytes'])
        if storage['required_inodes'] > os.statvfs(self.root).f_favail:
            raise IncompleteFreeze(['storage_inodes'])
        tasks = frozen_task_records(campaign)
        if [t['identity'] for t in tasks] != snapshot['tasks']:
            raise ValueError('validated freeze task roster differs')
        return campaign, runtime, tasks

    def publish(self, campaign, runtime, tasks):
        from oeis_learn.pipeline_validation.progressive_evidence import publish
        self.stage.mkdir(mode=0o700)
        atomic_write(self.stage / 'freeze-binding.json', self.binding)
        atomic_write(self.stage / 'owner.json', load(self.output / 'owner.json'))
        files = []
        for name, value in [('campaign.json', campaign), ('runtime.json', runtime)]:
            atomic_write(self.stage / name, value)
            files.append(reference(self.stage, name))
        (self.stage / 'tasks').mkdir()
        for task in tasks:
            name = 'tasks/' + task['identity'][7:] + '.json'
            atomic_write(self.stage / name, task)
            files.append(reference(self.stage, name))
        atomic_write(self.stage / 'freeze-publication.json', seal(dict(
            version='freeze-publication/v1', binding=self.binding['identity'], campaign=campaign['identity'], files=files)))
        result = publish(self.stage, operation='freeze', mechanical_status='COMPLETE',
                         reason='FROZEN_EXECUTION_STILL_REQUIRES_CURRENT_ACCEPTANCE', campaign=campaign['identity'])
        sync_directory(self.private)
        self.promote()
        return result
