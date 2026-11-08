#!/usr/bin/env python3
"""Serial focused T072 container test; print bounded evidence, never upload a ZIP."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.pipeline_validation.identity import digest,file_hash,seal
from oeis_learn.pipeline_validation.generalization_crash_proof import verify


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['podman','docker'],required=True)
    parser.add_argument('--output',default='run-state/generalization/pac-012-t072-001')
    args=parser.parse_args(argv)
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():parser.error('fresh output within repository required')
    source=digest(collect_snapshot(ROOT))
    command=[sys.executable,'-S',str(ROOT/'scripts/foundation_run.py'),'--engine',args.engine,
        'validate','--suite','pac-crash-hip','--output',str(output.relative_to(ROOT))]
    print('T072: building current pinned image, then serial HIP environment and crash/recovery proof',flush=True)
    result=subprocess.run(command,cwd=ROOT,check=False)
    if result.returncode:
        print(json.dumps(dict(status='FAIL',exit_code=result.returncode,output=str(output.relative_to(ROOT)))))
        return result.returncode
    stage=output/'pac-crash-hip-regressions'
    path=stage/'pac-crash-proof.json'
    if not path.is_file() or path.stat().st_size>256<<10:raise ValueError('bounded original crash proof missing')
    proof=verify(json.loads(path.read_text()),source=source)
    validation=json.loads((output/'validation.json').read_text())
    receipt=json.loads((stage/'receipt.json').read_text())
    if (validation['suite']!='pac-crash-hip' or validation['results']['pac-crash-hip-regressions']!=receipt
        or receipt['status']!='PASS' or receipt['counts']!={'tests':1,'skipped':0}
        or receipt['files'].get('pac-crash-proof.json')!=file_hash(path)):
        raise ValueError('original test/receipt/hash closure mismatch')
    summary=seal(dict(version='pac-t072-summary/v1',status='PASS',diagnostic_only=True,engine=args.engine,
        image=validation['image'],source=source,helper_sha256=file_hash(Path(__file__)),
        proof_sha256=file_hash(path),proof=proof,tests=1,skipped=0,zip_upload='NOT_REQUESTED'))
    target=output/'t072-summary.json'
    target.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('T072 result (paste this JSON or attach t072-summary.json; no ZIP needed):',flush=True)
    print(json.dumps(summary,sort_keys=True,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
