#!/usr/bin/env python3
"""Temporary serial HIP follow-up runner. Remove after target evidence review."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_command(command,root,log_path):
    """Retain the full log and expose failures without a wrapper traceback."""
    print('Command:',json.dumps(command),flush=True)
    print('Log:',log_path,flush=True)
    with log_path.open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=False)
    if result.returncode:
        print('Command failed with exit status',result.returncode,file=sys.stderr)
        with log_path.open('rb') as log:
            log.seek(0,2)
            log.seek(max(0,log.tell()-65536))
            tail=log.read().decode('utf-8',errors='replace').splitlines()[-100:]
        print('\n'.join(tail),file=sys.stderr)
        print('Full log:',log_path,file=sys.stderr)
    return result.returncode


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
    parser.add_argument('--start-step',type=int,default=1,help='start at this command number in a new output directory; invocations still requalify current source')
    args=parser.parse_args(argv)
    if not all(0<=seed<2**32 for seed in args.seeds):parser.error('seed outside 0..2**32-1')
    if len(set(args.seeds))!=len(args.seeds):parser.error('seed roster must be unique')
    root=Path(__file__).resolve().parents[1]
    for name in ('run','admission','inputs','output'):
        value=Path(getattr(args,name))
        if value.is_absolute() or '..' in value.parts:parser.error('workspace-relative paths required')
    queue=commands(args)
    if not 1<=args.start_step<=len(queue):parser.error('start-step outside command roster')
    if args.dry_run:
        print(json.dumps(queue[args.start_step-1:],indent=2));return 0
    target=root/args.output
    target.mkdir(parents=True,exist_ok=False)
    (target/'commands.json').write_text(json.dumps(queue,indent=2)+'\n')
    (target/'execution-plan.json').write_text(json.dumps(dict(start_step=args.start_step,skipped_steps=list(range(1,args.start_step)),
        interpretation='earlier commands omitted by operator; no qualification receipt reused'),indent=2)+'\n')
    for i,command in enumerate(queue):
        if i+1<args.start_step:continue
        print('Running',i+1,'of',len(queue),flush=True)
        code=run_command(command,root,target/(str(i+1)+'.log'))
        if code:
            (target/'failure.json').write_text(json.dumps(dict(step=i+1,command=command,returncode=code,
                log=str(i+1)+'.log'),indent=2)+'\n')
            return code if code>0 else 128-code
    return 0

if __name__=='__main__':raise SystemExit(main())
