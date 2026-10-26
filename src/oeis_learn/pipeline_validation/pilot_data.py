"""Finite larger generic corpus, preserving the historical512-row input format."""
from collections import Counter,defaultdict
import json
import sys
from pathlib import Path
import time
from .identity import seal,digest,file_hash,write,validate,confined
from .pilot_contract import SPEC,PROFILE
CHECKPOINTS=(65536,131072,262144)


_GENERIC_INDICES=list(map(str,range(100)))


def compact_reference(row):
    """Share equal immutable strings and the fixed index vector; bytes stay exact."""
    if row['indices']!=_GENERIC_INDICES:raise ValueError('fixed complete generic reference horizon required')
    row['indices']=_GENERIC_INDICES
    row['values']=[sys.intern(v) for v in row['values']]
    for key in ('source','parameter_source','family','generator','origin','stage','oracle','resource'):
        if key in row:row[key]=sys.intern(row[key])
    return row


def labels(row):
    from oeis_learn.fresh.bigint_splits import behavior,paired_tokens
    from oeis_learn.fresh.bigint_diagnostics import strata
    visible=dict(row,values=row['values'][:20])
    if 'parameter_source' in row:visible['tokens']=paired_tokens(row)['concrete']
    result=strata(visible)
    if 'parameter_source' not in row:result.update(length='unknown',complexity='unknown')
    constant,offset,nonaffine=behavior(row)
    return dict(result,stage=row.get('stage','OEIS'),nonaffine=str(nonaffine),offset_only=str(offset),constant=str(constant))


def learner_view(examples):
    if set(examples)!={'training','parameter_development','structural_development'}:
        raise ValueError('only generic visible learner splits permitted')
    out={};seen=set()
    allowed={'id','family','stage','indices','values','tokens','source','origin','horizon_indices'}
    for split,rows in examples.items():
        out[split]=[]
        for row in rows:
            if set(row)-allowed or len(row['indices'])!=20 or len(row['values'])!=20:
                raise ValueError('hidden continuation or unknown learner field')
            if row['id'] in seen:raise ValueError('duplicate cross-split learner identity')
            seen.add(row['id']);out[split].append(dict(row))
    return out


def choose(rows,*,blocked=(),deadline=None,progress=None):
    from .pilot_selection import select
    result=select(rows,training=SPEC['training'],parameter=SPEC['parameter_development'],
        structural=SPEC['structural_development'],blocked=blocked,deadline=deadline,progress=progress)
    if {r['stage'] for r in result['splits']['training']}!={'A','B','C','D'}:
        raise ValueError('generic stage coverage missing')
    return result


def publish(rows,oeis_prompts,oeis_truth,output,*,source,recipe,sampled,exclusions,admissions,lineage):
    from oeis_learn.fresh.bigint_inputs import visible_record
    from oeis_learn.fresh.bigint_splits import overlap_groups,diverse_tiny,behavior
    splits=rows['splits'];flat=[r for group in splits.values() for r in group]
    for r in oeis_prompts:
        t=oeis_truth[r['id']]
        if r['indices']!=t['indices'][:20] or r['values']!=t['values'][:20] or r['horizon_indices']!=t['indices']:
            raise ValueError('approved OEIS visible/hidden binding changed')
    oeis=[dict(r,family=r.get('family',r['id']),stage='OEIS',indices=oeis_truth[r['id']]['indices'],values=oeis_truth[r['id']]['values']) for r in oeis_prompts]
    if len(oeis)!=8 or len({r['id'] for r in oeis})!=8:raise ValueError('full frozen eight-row exposed OEIS-development required')
    groups=overlap_groups(flat+oeis)
    owner={}
    for name,rs in [*splits.items(),('oeis_development',oeis)]:
        for r in rs:
            group=groups[r['id']]
            if group in owner and owner[group]!=name:raise ValueError('cross-split prefix/shift overlap')
            owner[group]=name
    if {r['family'] for r in splits['training']}&{r['family'] for r in splits['structural_development']}:
        raise ValueError('structural family leak')
    examples=learner_view({k:[visible_record(r,'concrete') for r in rs] for k,rs in splits.items()})
    panels=diverse_tiny(splits['training'])
    evaluation=dict(training_panel=panels,parameter_development=splits['parameter_development'],
        structural_development=splits['structural_development'],oeis_development=oeis)
    prompts=[];truth={};roster={}
    for split,rs in evaluation.items():
        roster[split]=[]
        for r in rs:
            roster[split].append(r['id'])
            prompts.append(dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20],
                horizon_indices=r['indices'],family=r['family'],strata=dict(split=split,**labels(r))))
            truth[r['id']]=dict(indices=r['indices'],values=r['values'])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write(out/'examples.json',examples);write(out/'prompts.json',prompts);write(out/'private/truth.json',truth)
    write(out/'recipe.json',recipe);write(out/'private/admissions.json',admissions)
    tokens={k:sum(len(r['tokens']) for r in rs) for k,rs in examples.items()}
    audit=dict(counts={k:len(rs) for k,rs in examples.items()},tokens=tokens,selection=rows['selection_audit'],
        exposed_development=lineage,
        stages=dict(Counter(r['stage'] for r in splits['training'])),
        nonaffine=sum(behavior(r)[2] for r in splits['training']),sampled=sampled,exclusions=dict(exclusions),
        groups_identity=digest(groups),native_admissions=len(admissions),admissions_identity=digest(admissions),structural_families=len({r['family'] for r in splits['structural_development']}),
        training_families=len({r['family'] for r in splits['training']}),
        limitations='Observed100-index/20-window grouping, not universal equivalence; OEIS development already exposed.')
    from oeis_learn.fresh.bigint_splits import paired_tokens
    audit['splits']={}
    for split,rs in splits.items():
        lengths=[len(paired_tokens(r)['concrete']) for r in rs]
        strata_rows=[labels(r) for r in rs]
        audit['splits'][split]=dict(unique_sources=len({digest(r['source']) for r in rs}),
            unique_groups=len({groups[r['id']] for r in rs}),families=len({r['family'] for r in rs}),
            source_identity=digest([digest(r['source']) for r in rs]),assignment_identity=digest([r['assignment'] for r in rs]),
            target_lengths=dict(min=min(lengths),max=max(lengths),sum=sum(lengths)),
            parameter_counts=dict(Counter(len(r['assignment']) for r in rs)),
            strata={key:dict(Counter(r[key] for r in strata_rows)) for key in ('size','length','complexity','nonaffine','offset_only','constant')})
    m=seal(dict(version='pipeline-pilot-inputs/v2',source=source,profile=PROFILE['identity'],recipe=digest(recipe),
        roster=roster,audit=audit,files={name:{'sha256':file_hash(out/name),'bytes':(out/name).stat().st_size}
            for name in ('examples.json','prompts.json','private/truth.json','recipe.json','private/admissions.json')}))
    write(out/'prepared.json',m);return m


def check(root,*,hidden=False):
    root=Path(root);m=json.loads((root/'prepared.json').read_text())
    if m.get('version') not in ('pipeline-pilot-inputs/v1','pipeline-pilot-inputs/v2'):raise ValueError('registered pilot inputs required')
    validate(m,{'version','source','profile','recipe','roster','audit','files'},m['version'])
    if m['version']=='pipeline-pilot-inputs/v2':
        from .pilot_selection import POLICY
        selection=m['audit'].get('selection',{})
        if selection.get('policy')!=POLICY or selection.get('counts')!=m['audit']['counts'] or selection.get('required')!=m['audit']['counts']:
            raise ValueError('registered complete selected-population policy required')
    if m['profile']!=PROFILE['identity'] or m['audit']['counts']!={k:SPEC[k] for k in ('training','parameter_development','structural_development')}:
        raise ValueError('full registered pilot corpus required')
    if {k:len(v) for k,v in m['roster'].items()}!={'training_panel':32,'parameter_development':128,'structural_development':128,'oeis_development':8}:
        raise ValueError('complete frozen evaluation roster required')
    if len({v for ids in m['roster'].values() for v in ids})!=296:raise ValueError('duplicate evaluation prompt')
    if set(m['files'])!={'examples.json','prompts.json','private/truth.json','recipe.json','private/admissions.json'}:raise ValueError('closed pilot file inventory required')
    from oeis_learn.synthesis.state import require_hash
    require_hash(m['source'])
    if m['audit']['native_admissions']!=33024:raise ValueError('all selected generic sources require independent production admission')
    if digest(json.loads((root/'recipe.json').read_text()))!=m['recipe']:raise ValueError('pilot recipe binding changed')
    for name,r in m['files'].items():
        if name.startswith('private/') and not hidden:continue
        f=confined(root,name)
        if f.stat().st_size!=r['bytes'] or file_hash(f)!=r['sha256']:raise ValueError('prepared pilot file mismatch')
    if hidden:
        admissions=json.loads((root/'private/admissions.json').read_text())
        if len(admissions)!=33024 or digest(admissions)!=m['audit']['admissions_identity']:raise ValueError('pilot production admission inventory changed')
        truth=json.loads((root/'private/truth.json').read_text())
        if set(truth)!={v for ids in m['roster'].values() for v in ids}:raise ValueError('independent scoring truth roster changed')
        for row in truth.values():
            if set(row)!={'indices','values'} or len(row['indices'])!=100 or len(row['values'])!=100:raise ValueError('exact100-index truth required')
    return m


def prepare(inputs,output,config):
    from dataclasses import asdict
    from oeis_learn.data.bigint_programs import BigintSampler,materialize
    from oeis_learn.synthesis.contracts import ResourceProfile,canonical_integer
    from oeis_learn.synthesis.client import SynthesisClient
    from .pilot_recipe import recipe
    from .pilot_cache import ReferenceCache,binding
    from oeis_learn.fresh.common import write as replace_json
    opt=config['options'];root=Path(output)/'private/prepared'
    begin=time.monotonic();deadline=begin+opt['seconds']-60
    prompts=json.loads((inputs/'oeis-prompts.json').read_text());truth=json.loads((inputs/'oeis-truth.json').read_text())
    lineage=json.loads((inputs/'oeis-lineage.json').read_text())
    validate(lineage,{'version','prepared','files','count','exposure'},'pipeline-pilot-exposed-development/v1')
    if lineage['count']!=8 or set(lineage['files'])!={'oeis-prompts.json','oeis-truth.json'}:raise ValueError('frozen exposed-development lineage required')
    for name,ref in lineage['files'].items():
        if file_hash(inputs/name)!=ref['sha256'] or (inputs/name).stat().st_size!=ref['bytes']:raise ValueError('exposed-development lineage changed')
    sampler=BigintSampler(SPEC['seed']);limits=ResourceProfile();rows=[];excluded=Counter();selected=None
    cache=ReferenceCache(Path(output)/'private/reference-cache',binding(sampler,limits),
        seed=inputs/'reference-cache' if (inputs/'reference-cache/index.json').exists() else None)
    cached=cache.count;records=iter(cache.records());stats=dict(version='pipeline-pilot-preparation/v2',source=config['source'],
        phase='sampling',status='RUNNING',sampled=0,eligible=0,available_reference_counters=cached,reused_reference_counters=0,selection_attempts=[],exclusions={})
    def report():
        stats.update(elapsed_seconds=time.monotonic()-begin,eligible=len(rows),exclusions=dict(excluded),
            reference_cache=dict(binding=cache.expected,count=cache.count,bytes=cache.value['bytes'],index_sha256=file_hash(cache.path)))
        replace_json(Path(output)/'preparation.json',stats)
    report()
    try:
        if cached:
            provenance_path=inputs/'reference-provenance.json'
            if provenance_path.stat().st_size>32768:raise ValueError('bounded reference provenance required')
            provenance=json.loads(provenance_path.read_text())
            validate(provenance,{'version','binding','index_sha256','producer','receipt','state','origin','authority'},'pipeline-pilot-reference-provenance/v1')
            if (provenance['binding']!=cache.expected or provenance['index_sha256']!=file_hash(cache.path)
                or provenance['authority']!='RAW_REFERENCE_ONLY'):raise ValueError('reference provenance/cache binding changed')
            stats['reference_provenance']=provenance;report()
        for counter in range(SPEC['max_source_counters']):
            if time.monotonic()>=deadline:raise TimeoutError('finite pilot corpus preparation deadline')
            if counter<cached:
                record=next(records);row=record['row']
                stats['reused_reference_counters']+=1
                if row is not None:
                    expected=sampler.sample(counter)
                    if set(row)!=set(expected)|{'indices','values','oracle','resource'}:
                        raise ValueError('closed cached reference fields required')
                    if row['oracle']!='independent-source-reference/v1':raise ValueError('cached reference oracle changed')
                    if any(row.get(k)!=v for k,v in expected.items()) or row.get('resource')!=digest(asdict(limits)):
                        raise ValueError('cached reference counter/source/resources changed')
                    if row['indices']!=list(map(str,range(100))) or len(row['values'])!=100:
                        raise ValueError('complete cached100-index reference required')
                    for i,v in enumerate(row['values']):
                        n=canonical_integer(v,bits=limits.intermediate_bits)
                        if i<20 and abs(n).bit_length()>limits.input_bits:raise ValueError('cached reference input integer limit')
                else:excluded[record['exclusion']]+=1
            else:
                try:row=materialize(sampler.sample(counter),limits)
                except ValueError as exc:
                    excluded[str(exc)]+=1;row=None;cache.append(counter,None,str(exc))
                else:cache.append(counter,row)
            if row is not None:rows.append(compact_reference(row))
            stats['sampled']=counter+1
            if (counter+1)%256==0:
                replace_json(Path(output)/'progress.json',dict(counter=counter+1,eligible=len(rows),seconds=time.monotonic()-begin));report()
            if counter+1 not in CHECKPOINTS:continue
            stats['phase']='selection';cache.flush();report()
            try:
                oeis=[dict(id=r['id'],indices=truth[r['id']]['indices'],values=truth[r['id']]['values']) for r in prompts]
                def selection_progress(audit):
                    stats['selection_progress']=audit;report()
                selected=choose(rows,blocked=oeis,deadline=deadline,progress=selection_progress)
                stats['selection_attempts'].append(dict(counter=counter+1,status='COMPLETE',audit=selected['selection_audit']));break
            except ValueError as exc:
                excluded[str(exc)]+=1
                stats['selection_attempts'].append(dict(counter=counter+1,status='SHORTFALL',error=str(exc),audit=getattr(exc,'audit',None)))
                stats['phase']='sampling';report()
        if selected is None:
            raise ValueError('full pilot corpus shortfall; no padding/fallback: '+str(stats['selection_attempts']))
        stats['phase']='native-admission';report();admitted=0;admissions={}
        with SynthesisClient(digest(PROFILE),limits=limits,cache_root=Path(output)/'private/admission-cache') as client:
            for rs in selected['splits'].values():
                for r in rs:
                    if time.monotonic()>=deadline:raise TimeoutError('complete native admission deadline')
                    result=client.execute(r['source'],r['indices'],deadline=min(deadline,time.monotonic()+30))
                    if result.get('outcome')!='ok' or result.get('values')!=r['values']:
                        stats['native_failure']=dict(id=r['id'],source=digest(r['source']),outcome=result.get('outcome'),reason=result.get('reason'),coverage=len(result.get('values',[])))
                        raise ValueError('selected generic source lacks complete independent production agreement: '+str(stats['native_failure']))
                    admitted+=1;stats['native_admitted']=admitted
                    admissions[r['id']]=dict(source=digest(r['source']),horizon=digest(dict(indices=r['indices'],values=r['values'])),resource=r['resource'],runtime=client.runtime_identity('bigint-body/v1'),independent_reference=True)
                    if admitted%256==0:
                        replace_json(Path(output)/'progress.json',dict(admitted=admitted,seconds=time.monotonic()-begin));report()
        stats['phase']='publication';report()
        m=publish(selected,prompts,truth,root,source=config['source'],recipe=recipe(),sampled=counter+1,exclusions=excluded,admissions=admissions,lineage=lineage)
        check(root,hidden=True);write(Path(output)/'corpus-audit.json',m['audit']);stats.update(status='COMPLETE',phase='complete')
        return dict(status='COMPLETE',work=m['identity'],prepared='private/prepared',counts=m['audit']['counts'],native_admitted=admitted,seconds=time.monotonic()-begin)
    except BaseException as exc:
        stats.update(status='FAILED',error=type(exc).__name__+': '+str(exc));raise
    finally:
        cache.flush();report()
