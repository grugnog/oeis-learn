#!/usr/bin/env python3
"""Temporary serial HIP follow-up runner. Remove after target evidence review."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(args):
    base=[sys.executable,'scripts/foundation_run.py','--engine',args.engine]
    common=['--run',args.run,'--admission',args.admission,'--inputs',args.inputs]
    out=Path(args.output)
    result=[base+['validate','--suite',suite,'--output',str(out/suite)] for suite in ('bigint-cpu','bigint-gpu')]
    result.append(base+['pilot-diagnose','models',*common,'--snapshots','2','5','--include-selected','--output',str(out/'models')])
    for name in ('matched','matched-drain'):
        result.append(base+['pilot-diagnose','evaluate',*common,'--snapshots','2','5','--include-selected',
            '--evaluation-policy','portfolio-clock60-'+name+'/v2','--output',str(out/name)])
    if not args.skip_training:
        for seed in args.seeds:
            result.append(base+['pilot-followup',*common,'--tokens',str(args.tokens),'--seed',str(seed),
                '--output',str(out/('study-'+str(seed)))])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['docker','podman'],default='podman')
    for name in ('run','admission','inputs','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--tokens',type=int,choices=[25000000,50000000],default=25000000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261016])
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds):parser.error('seed roster must be unique')
    root=Path(__file__).resolve().parents[1]
    for name in ('run','admission','inputs','output'):
        value=Path(getattr(args,name))
        if value.is_absolute() or '..' in value.parts:parser.error('workspace-relative paths required')
    queue=commands(args)
    if args.dry_run:
        print(json.dumps(queue,indent=2));return 0
    target=root/args.output
    target.mkdir(parents=True,exist_ok=False)
    (target/'commands.json').write_text(json.dumps(queue,indent=2)+'\n')
    for i,command in enumerate(queue):
        print('Running',i+1,'of',len(queue),flush=True)
        with (target/(str(i+1)+'.log')).open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
