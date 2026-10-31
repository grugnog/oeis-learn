#!/usr/bin/env python3
"""Serial canonical acceptance driver; no research, acceptance, or reuse decisions."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from oeis_learn.pipeline_validation.identity import confined,seal
from oeis_learn.fresh.common import write
from oeis_learn.generalization.resources import reserve_stage,log_tail

CASES=('retained-eval','refresh','parameter-targets','replay-resume','failure-packaging')


def plan(engine,manifest,output,*,reuse_run=None,image=None):
    output=Path(output)
    base=[sys.executable,str(ROOT/'scripts/foundation_run.py'),'--engine',engine]
    if image is not None:base+=['--image',image]
    base+=['generalization']
    stages=[]
    def add(name,operation,seconds,category,extra=()):
        argv=base+[operation,'--output',str(output/name)]
        if operation!='verify-bundle':
            argv+=['--manifest',manifest]
            if reuse_run is not None:argv+=['--reuse-run',reuse_run]
        argv+=list(extra)
        stages.append(dict(name=name,operation=operation,seconds=seconds,category=category,argv=argv))
    add('preflight','preflight',600,'preflight')
    add('qualify','qualify',7200,'qualification')
    qualification=['--qualification',str(output/'qualify')]
    add('prepare','prepare',600,'preparation',qualification)
    for case in CASES:
        add(case,'acceptance',900,'cases',qualification+['--case',case,'--inputs',str(output/'prepare')])
    add('report','report',600,'packaging',['--inputs',str(output)])
    add('verify-bundle','verify-bundle',300,'packaging',['--archive',str(output/'report.zip')])
    return stages


def run(stage,root,seconds):
    from oeis_learn.pipeline_validation.supervision import launch
    logs=root/'private/logs'/stage['name']
    try:
        from oeis_learn.pipeline_validation.generalization_lifecycle import guard
        with guard(logs/'disk'):
            launch(stage['argv'],logs,seconds=seconds,stream_cap=1<<20)
        return 0
    except KeyboardInterrupt:return 130
    except (OSError,RuntimeError,TimeoutError):
        process=logs/'process.json'
        code=json.loads(process.read_text()).get('returncode') if process.exists() else None
        return code if code in (1,2,3,130) else 1
    finally:
        for channel in ('stdout','stderr'):
            path=logs/(channel+'.log')
            if path.is_file():
                target=root/'logs'/(stage['name']+'-'+channel+'.txt')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(log_tail(path.read_text(errors='replace')))


def recover(root,ledger):
    """Minimal existing archive format, explicitly not a qualified feature bundle."""
    from oeis_learn.pipeline_validation.bundle import pack,verify
    state='INTERRUPTED' if any(r['exit_code']==130 for r in ledger['receipts']) else 'FAIL'
    write(root/'summary.json',seal(dict(selected_scope=state,reason='PACKAGE_FAILED',
        scientific_status='UNMEASURED',feature_qualified=False,stages={})))
    import shutil
    recovery=root/'recovery';recovery.mkdir(exist_ok=False)
    for name in ('summary.json','workflow.json'):
        shutil.copyfile(root/name,recovery/name)
    if (root/'logs').is_dir():shutil.copytree(root/'logs',recovery/'logs')
    archive=pack(recovery,mandatory=['summary.json','workflow.json'],optional_cap=8<<20)
    target=root.with_suffix('.zip')
    if target.exists():raise ValueError('recovery archive already exists')
    archive.rename(target);archive=target
    verify(archive)
    return archive


def execute(stages,root,*,runner=run):
    root=Path(root)
    if root.exists() or root.with_suffix('.zip').exists():raise ValueError('fresh workflow root and archive required')
    root.mkdir(parents=True,exist_ok=False)
    ledger=dict(version='generalization-serial-workflow/v1',plan=stages,receipts=[],
                research_authorized=False,status='PARTIAL')
    started=time.monotonic();failure=0;report_ok=False;verified=False
    write(root/'workflow.json',seal(ledger))
    for stage in stages:
        packaging=stage['category']=='packaging'
        if failure and not packaging:continue
        if stage['operation']=='verify-bundle' and (not report_ok or not (root/'report.zip').is_file()):continue
        elapsed=int(time.monotonic()-started)
        try:
            reserve_stage(stage['category'],elapsed_seconds=elapsed,requested_seconds=stage['seconds'])
            code=runner(stage,root,stage['seconds'])
        except KeyboardInterrupt:code=130
        except (OSError,RuntimeError,ValueError,TimeoutError):code=1
        if code not in (0,1,2,3,130):code=1
        ledger['receipts'].append(dict(stage=stage['name'],exit_code=code,
            elapsed_seconds=time.monotonic()-started))
        if code and not failure:failure=code
        if stage['operation']=='report':report_ok=(root/'report.zip').is_file()
        if stage['operation']=='verify-bundle':
            summary=root/'verify-bundle/summary.json'
            verified=code==0 or summary.is_file() and json.loads(summary.read_text()).get('integrity')=='PASS'
        ledger['status']='INTERRUPTED' if failure==130 else 'FAIL' if failure in (1,2) else 'BLOCKED' if failure else 'PARTIAL'
        write(root/'workflow.json',seal(ledger))
    if not verified:
        ledger['status']='INTERRUPTED' if failure==130 else 'FAIL'
        ledger['reason']='PACKAGE_FAILED'
        write(root/'workflow.json',seal(ledger))
        recover(root,ledger)
        return failure or 1
    ledger['status']='PASS' if not failure else ledger['status']
    write(root/'workflow.json',seal(ledger))
    import shutil
    shutil.copyfile(root/'report.zip',root.with_suffix('.zip'))
    return failure


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('docker','podman'),required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-run')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    try:
        output=confined(ROOT,args.output,exists=False)
        confined(ROOT,args.manifest)
        if args.reuse_run:
            previous=confined(ROOT,args.reuse_run)
            if previous==output:raise ValueError('workflow restart requires a new root')
        from oeis_learn.pipeline_validation.generalization_session import load_session
        _,children=load_session(ROOT,args.manifest)
        image=children['refresh'][2]['images'][0]
        stages=plan(args.engine,args.manifest,args.output,reuse_run=args.reuse_run,image=image)
        if args.dry_run:
            print(json.dumps(dict(stages=stages,ceiling_seconds=14400,margin_seconds=600,
                research_authorized=False,reuse='canonical endpoints verify original producers'),indent=2))
            return 0
        from oeis_learn.generalization.resources import CAPS
        from oeis_learn.pipeline_validation.generalization_lifecycle import scope
        limits={k:min(CAPS[k],sum(c[2]['storage'][k] for c in children.values())) for k in CAPS}
        with scope(output,limits,extra_roots=[c[0] for c in children.values()]):return execute(stages,output)
    except (OSError,ValueError) as exc:
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
