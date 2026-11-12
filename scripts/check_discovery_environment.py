#!/usr/bin/env python3
"""Check the selected checkout and workstation; engine probes use Podman/Docker.

The launcher stays stdlib-only so even missing repository modules get a ZIP.
It does not build images, launch training, use the GPU or modify existing runs.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid
import zipfile


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    options, _ = parser.parse_known_args()
    root = Path(options.root).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'src'))
    try:
        from oeis_learn.pipeline_validation.environment_preflight import main as inspect
    except Exception as exc:
        # The ordinary collector is unavailable. Preserve the exact bootstrap
        # failure without requiring any package import or touching prior evidence.
        out = root / 'run-state' / ('environment-preflight-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        try:
            out.mkdir(parents=True, exist_ok=False)
        except OSError:
            out = Path(tempfile.mkdtemp(prefix='oeis-preflight-'))
        report = dict(version='discovery-environment-preflight/v1', root=str(root),
                      python=sys.version, counts=dict(PASS=0, WARN=0, FAIL=1),
                      checks=[dict(name='checkout bootstrap', status='FAIL',
                                   error=type(exc).__name__, detail=str(exc),
                                   traceback=traceback.format_exc())],
                      guidance='Use the spec 013 branch and git pull --ff-only; do not install host ML dependencies to repair missing source files.')
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        archive = out / 'preflight.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(out / 'report.json', 'report.json')
            z.write(Path(__file__).resolve(), 'check_discovery_environment.py')
        print('FAIL checkout bootstrap: ' + str(exc), flush=True)
        print('Share: ' + str(archive), flush=True)
        return 1
    return inspect(default_root=root)


if __name__ == '__main__':
    raise SystemExit(main())
