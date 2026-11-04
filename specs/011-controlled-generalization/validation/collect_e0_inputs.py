#!/usr/bin/env python3
"""Bounded, no-engine discovery of the exact already-exposed E0 input files."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
METADATA = ('import.json', 'best-checkpoint.json', 'latest-checkpoint.json',
            'best-provenance.json', 'latest-provenance.json',
            *(f'provenance/{name}.json' for name in
              ('report', 'producer', 'history', 'prepared', 'recipe', 'training')))
SKIP = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'target',
        '.runtime', '.cache', '.uv-cache', 'container-cache', 'checkpoints'}


def read(path, root, cap=4 << 20):
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular input required: ' + str(path))
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    with path.open('rb') as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError('input byte limit: ' + str(path))
    return data


def matches(path, root, ref):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != ref['bytes']:
        return None
    data = read(path, root)
    return data if 'sha256:' + hashlib.sha256(data).hexdigest() == ref['sha256'] else None


def collect(workspace, imported, output, *, search_roots=None,
            max_entries=100000, seconds=60):
    from oeis_learn.generalization.legacy import verify_legacy
    from oeis_learn.pipeline_validation.identity import seal
    workspace = Path(workspace).resolve(strict=True)
    imported = workspace / imported
    output = workspace / output
    output.resolve(strict=False).relative_to(workspace)
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output required')
    payloads = {name: read(imported / name, workspace) for name in METADATA}
    roster = json.loads(payloads['import.json'])
    descriptors = [json.loads(payloads[label + '-checkpoint.json']) for label in ('best', 'latest')]
    if (seal(roster) != roster or roster.get('version') != 'generalization-legacy-roster/v1'
            or roster.get('checkpoints') != descriptors):
        raise ValueError('original uniform import roster required')
    for descriptor in descriptors:
        verify_legacy(imported, descriptor)
    prepared = json.loads(payloads['provenance/prepared.json'])
    # The legacy validator above checks the original prepared metadata binding.
    # Only the historical exposed prompt/truth inventory is eligible for export.
    expected = {name: prepared['files'][name] for name in ('prompts.json', 'private/truth.json')}
    deadline = time.monotonic() + seconds
    queue = deque()
    for name in search_roots or ['.']:
        path = workspace / name
        if path.is_symlink():
            raise ValueError('search root cannot be a symlink')
        path.resolve(strict=True).relative_to(workspace)
        queue.append((path, 0))
    visited = set()
    entries = 0
    found = None
    while queue and found is None:
        folder, depth = queue.popleft()
        if folder in visited:
            continue
        visited.add(folder)
        if time.monotonic() >= deadline:
            raise ValueError('search time bound reached; retry with an explicit --search-root')
        with os.scandir(folder) as iterator:
            for item in iterator:
                entries += 1
                if entries > max_entries or time.monotonic() >= deadline:
                    raise ValueError('search bound reached; retry with an explicit --search-root')
                if item.is_symlink():
                    continue
                if item.is_dir(follow_symlinks=False):
                    if depth < 12 and item.name not in SKIP:
                        queue.append((Path(item.path), depth + 1))
                    continue
                if item.name != 'prompts.json':
                    continue
                prompt = matches(Path(item.path), workspace, expected['prompts.json'])
                if prompt is None:
                    continue
                # Require the original cohort metadata beside both matched files.
                metadata = folder / 'prepared.json'
                if not metadata.is_file() or read(metadata, workspace) != payloads['provenance/prepared.json']:
                    continue
                truth = matches(folder / 'private/truth.json', workspace, expected['private/truth.json'])
                if truth is not None:
                    found = (folder, prompt, truth)
                    break
    if found is None:
        raise ValueError('exact original prompts/truth/prepared triple not found; no archive created')
    folder, prompt, truth = found
    # This package explicitly includes historical truth, not any new sealed cohort.
    payloads.update({'historical/prompts.json': prompt, 'historical/truth.json': truth})
    record = dict(version='e0-historical-input-discovery/v1',
                  purpose='already-exposed historical evaluation inputs; not qualification',
                  prepared=prepared['identity'], source_root=str(folder.relative_to(workspace)),
                  checkpoints=[d['identity'] for d in descriptors],
                  members=[dict(path=name, bytes=len(data), sha256='sha256:' + hashlib.sha256(data).hexdigest())
                           for name, data in payloads.items()])
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr('discovery.json', json.dumps(record, indent=2))
    return dict(status='COLLECTED', archive=str(output.relative_to(workspace)),
                source_root=record['source_root'], files=len(payloads),
                qualification='NOT_RUN', hip='NOT_RUN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imported', default='run-state/generalization/uniform-import')
    parser.add_argument('--output', default='run-state/generalization/e0-historical-inputs-001.zip')
    parser.add_argument('--search-root', action='append')
    args = parser.parse_args()
    try:
        result = collect(ROOT, args.imported, args.output, search_roots=args.search_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
