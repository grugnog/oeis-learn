"""Stdlib bounded readers and non-mutating collection of existing pilot artifacts."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import stat
import zipfile
from .identity import confined,file_hash,seal,write


class Evidence:
    def __init__(self,path):
        from .bundle import verify
        from .reuse import validate_import
        self.path=Path(path);self.zip=None
        if self.path.is_symlink():raise ValueError('regular evidence root required')
        if self.path.is_dir():
            index=confined(self.path,'manifest.json')
            if index.stat().st_size>4<<20:raise ValueError('bounded evidence inventory required')
            provisional=json.loads(index.read_text())
            if len(provisional['members'])>4096 or sum(r['bytes'] for r in provisional['members'])>512<<20:raise ValueError('bounded public evidence required')
            self.manifest=validate_import(self.path)
        else:
            with zipfile.ZipFile(self.path) as archive:
                info=archive.infolist()
                if len(info)>4097 or sum(i.file_size for i in info)>512<<20:raise ValueError('bounded public archive required')
                if any(Path(i.filename).is_absolute() or '..' in Path(i.filename).parts or stat.S_ISLNK(i.external_attr>>16) for i in info):raise ValueError('unsafe public archive member')
            self.manifest=verify(self.path);self.zip=zipfile.ZipFile(self.path)
        self.inventory={r['path']:r for r in self.manifest['members']}
        if len(self.inventory)>4096 or sum(r['bytes'] for r in self.inventory.values())>512<<20:raise ValueError('bounded diagnostic evidence required')
        self.summary=self.read('summary.json');self.invocation=self.read('invocation.json')
        if seal(self.invocation)!=self.invocation:raise ValueError('changed producer invocation')
        for key,row in self.summary['stages'].items():
            if seal(row)!=row or row!=self.read('stages/'+key+'/receipt.json'):raise ValueError('changed producer stage')
            for name,sha in row['files'].items():
                if self.inventory.get('stages/'+key+'/'+name,{}).get('sha256')!=sha:raise ValueError('unbound producer file')
            for parent,sha in row.get('prerequisites',{}).items():
                p=self.summary['stages'].get(parent,{})
                if sha not in (p.get('identity'),p.get('reused_from')):raise ValueError('broken producer prerequisite')

    def bytes(self,name):
        if name not in self.inventory:raise ValueError('file absent from producer manifest: '+name)
        if self.inventory[name]['bytes']>64<<20:raise ValueError('individual public diagnostic input exceeds64MiB')
        if self.zip:return self.zip.read(name)
        p=confined(self.path,name)
        if not p.is_file():raise ValueError('regular evidence file required')
        return p.read_bytes()

    def read(self,name):return json.loads(self.bytes(name))
    def has(self,name):return name in self.inventory
    def close(self):
        if self.zip:self.zip.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def collect(reader,output):
    """Historical outcomes only; never reclassifies a timeout as success/failure."""
    output=Path(output);milestones=[];missing=[]
    for i in range(6):
        prefix=f'stages/pilot.evaluate-{i}/';name=prefix+f'snapshot-{i}.json'
        if not reader.has(name):missing.append(i);continue
        point=reader.read(name);gen=prefix+f'generate-{i}/';score=prefix+f'score-{i}/'
        records=reader.read(gen+'selections.json');outcomes=reader.read(score+'outcomes.json')
        if point['outcomes']!=outcomes or set(records)!=set(outcomes):raise ValueError('snapshot/scoring roster mismatch')
        by_group=defaultdict(Counter);diversity=defaultdict(set);lengths=defaultdict(list);modes=defaultdict(Counter)
        for pid,arms in outcomes.items():
            meta=records[pid]
            for arm,r in arms.items():
                if r['success'] and not r['complete']:raise ValueError('success lacks complete independent scoring')
                for group in ('all','split:'+meta['strata']['split'],'family:'+meta['family'],*[k+':'+str(v) for k,v in meta['strata'].items() if k!='split']):
                    c=by_group[group+'|'+arm];c['denominator']+=1;c['success']+=r['success'];c['complete']+=r['complete'];c['reason:'+str(r.get('reason'))]+=1
                selected=meta['selections'][arm]
                if selected:
                    s=selected['selected'];diversity[arm].add(s['source_sha256']);lengths[arm].append(len(s['source'].encode()))
        attempt_name=gen+'attempts.jsonl'
        if reader.has(attempt_name):
            for line in reader.bytes(attempt_name).splitlines():
                r=json.loads(line);c=modes[r['mode']];c['event:'+r['event']]+=1
                for key in ('tokens_count','generation_seconds','fit_seconds'):
                    value=r['payload'].get(key)
                    if type(value) in (int,float):c[key]+=value
        for c in by_group.values():
            c['unresolved']=c['denominator']-c['complete'];c['lower']=c['success']/c['denominator'];c['upper']=(c['success']+c['unresolved'])/c['denominator']
        train=f'stages/pilot.train-{i}/train-{i}/'
        history=reader.read(train+'validation-history.json') if reader.has(train+'validation-history.json') else []
        bins=reader.read(train+'training-bins.json') if reader.has(train+'training-bins.json') else []
        milestones.append(dict(ordinal=i,tokens=point['tokens'],updates=point['updates'],checkpoint=point['checkpoint'],
            groups=dict(by_group),attempt_event_aggregates=dict(modes),historical_evaluation_cost=point.get('evaluation_cost'),
            selected_program_diversity={a:dict(unique_sources=len(s),selected_count=len(lengths[a]),
                median_source_bytes=statistics.median(lengths[a])) for a,s in diversity.items()},
            teacher_forcing_history=history,training_bins=bins))
    # Keep existing resource summaries/times, avoiding duplicate raw sample streams.
    audits={n:reader.read(n) for n in reader.inventory if n.endswith(('/memory.json','/host-timing.json','/disk.json'))}
    result=seal(dict(version='pipeline-pilot-collected-diagnostics/v1',producer_source=reader.invocation['binding'].get('source'),
        producer_study=reader.invocation['binding'].get('study'),producer_states={k:r['state'] for k,r in reader.summary['stages'].items()},
        milestones=milestones,missing_milestones=missing,role_audits=audits,
        original_trace_limit='Compact events do not contain original full token streams, logits, activations or gradients.',
        interpretation='Historical finite-budget evidence; missing remains unresolved. No new semantic score, admission or causal claim.',
        training=False,automatic_run_authorized=False))
    from .pilot_runtime import bounded_write
    bounded_write(output/'collected.json',result,16<<20)
    return dict(available_milestones=len(milestones),missing_milestones=missing,producer_source=result['producer_source'])
