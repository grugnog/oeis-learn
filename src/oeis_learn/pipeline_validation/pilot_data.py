"""Finite larger generic corpus, preserving the historical512-row input format."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
from .identity import seal,digest,file_hash,write,validate,confined
from .pilot_contract import SPEC,PROFILE


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


def choose(rows):
    """Reuse old matched holdouts, then fill nonreserved families without duplicates."""
    from oeis_learn.fresh.bigint_splits import choose_splits,paired_tokens,behavior
    from oeis_learn.synthesis.contracts import identity
    result=choose_splits(rows);groups=result['groups'];splits=result['splits']
    eligible=[];variants=defaultdict(set)
    for r in rows:
        if r['id'] in groups:variants[r['family']].add(r['variant']);eligible.append(r)
    families=sorted((k for k,v in variants.items() if len(v)>=16),key=lambda k:identity(['structural-reservation/v1',k]))
    reserved=set(families[::4]);used={groups[r['id']] for rs in splits.values() for r in rs}
    pools=defaultdict(list)
    for r in sorted(eligible,key=lambda r:r['id']):
        if r['family'] not in reserved and r['family'] in families:pools[r['stage']].append(r)
    positions={k:0 for k in pools};training=splits['training'];facts=[behavior(r) for r in training]
    constant=sum(x[0] for x in facts);offset=sum(x[1] for x in facts)
    while len(training)<SPEC['training']:
        progress=False
        for stage in sorted(pools):
            while positions[stage]<len(pools[stage]):
                r=pools[stage][positions[stage]];positions[stage]+=1
                if groups[r['id']] in used:continue
                c,o,_=behavior(r)
                if constant+c>SPEC['training']//10 or offset+o>SPEC['training']//10:continue
                training.append(r);used.add(groups[r['id']]);constant+=c;offset+=o;progress=True;break
            if len(training)==SPEC['training']:break
        if not progress:raise ValueError(f'pilot unique training shortfall:{len(training)}/{SPEC["training"]}')
    if {r['stage'] for r in training}!={'A','B','C','D'}:raise ValueError('generic stage coverage missing')
    return result


def publish(rows,oeis_prompts,oeis_truth,output,*,source,recipe,sampled,exclusions):
    from oeis_learn.fresh.bigint_inputs import visible_record
    from oeis_learn.fresh.bigint_splits import overlap_groups,diverse_tiny,behavior
    splits=rows['splits'];flat=[r for group in splits.values() for r in group]
    oeis=[dict(r,indices=oeis_truth[r['id']]['indices'],values=oeis_truth[r['id']]['values']) for r in oeis_prompts]
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
                horizon_indices=r['indices'],family=r['family'],strata=dict(split=split,stage=r.get('stage','OEIS'))))
            truth[r['id']]=dict(indices=r['indices'],values=r['values'])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write(out/'examples.json',examples);write(out/'prompts.json',prompts);write(out/'private/truth.json',truth)
    write(out/'recipe.json',recipe)
    tokens={k:sum(len(r['tokens']) for r in rs) for k,rs in examples.items()}
    audit=dict(counts={k:len(rs) for k,rs in examples.items()},tokens=tokens,
        stages=dict(Counter(r['stage'] for r in splits['training'])),
        nonaffine=sum(behavior(r)[2] for r in splits['training']),sampled=sampled,exclusions=dict(exclusions),
        groups_identity=digest(groups),structural_families=len({r['family'] for r in splits['structural_development']}),
        training_families=len({r['family'] for r in splits['training']}),
        limitations='Observed100-index/20-window grouping, not universal equivalence; OEIS development already exposed.')
    m=seal(dict(version='pipeline-pilot-inputs/v1',source=source,profile=PROFILE['identity'],recipe=digest(recipe),
        roster=roster,audit=audit,files={name:{'sha256':file_hash(out/name),'bytes':(out/name).stat().st_size}
            for name in ('examples.json','prompts.json','private/truth.json','recipe.json')}))
    write(out/'prepared.json',m);return m


def check(root,*,hidden=False):
    root=Path(root);m=json.loads((root/'prepared.json').read_text())
    validate(m,{'version','source','profile','recipe','roster','audit','files'},'pipeline-pilot-inputs/v1')
    if m['profile']!=PROFILE['identity'] or m['audit']['counts']!={k:SPEC[k] for k in ('training','parameter_development','structural_development')}:
        raise ValueError('full registered pilot corpus required')
    if {k:len(v) for k,v in m['roster'].items()}!={'training_panel':32,'parameter_development':128,'structural_development':128,'oeis_development':8}:
        raise ValueError('complete frozen evaluation roster required')
    if len({v for ids in m['roster'].values() for v in ids})!=296:raise ValueError('duplicate evaluation prompt')
    for name,r in m['files'].items():
        if name=='private/truth.json' and not hidden:continue
        f=confined(root,name)
        if f.stat().st_size!=r['bytes'] or file_hash(f)!=r['sha256']:raise ValueError('prepared pilot file mismatch')
    return m


def prepare(inputs,output,config):
    from oeis_learn.data.bigint_programs import BigintSampler,materialize
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.client import SynthesisClient
    from .pilot_recipe import recipe
    from oeis_learn.fresh.common import write as replace_json
    opt=config['options'];root=Path(output)/'private/prepared'
    begin=time.monotonic();deadline=begin+opt['seconds']-60
    prompts=json.loads((inputs/'oeis-prompts.json').read_text());truth=json.loads((inputs/'oeis-truth.json').read_text())
    sampler=BigintSampler(SPEC['seed']);limits=ResourceProfile();rows=[];excluded=Counter();selected=None
    for counter in range(SPEC['max_source_counters']):
        if time.monotonic()>=deadline:raise TimeoutError('finite pilot corpus preparation deadline')
        try:rows.append(materialize(sampler.sample(counter),limits))
        except ValueError as exc:excluded[str(exc)]+=1
        if (counter+1)%256==0:replace_json(Path(output)/'progress.json',dict(counter=counter+1,eligible=len(rows),seconds=time.monotonic()-begin))
        if counter+1 not in (65536,131072,262144):continue
        try:selected=choose(rows);break
        except ValueError as exc:excluded[str(exc)]+=1
    if selected is None:raise ValueError('full pilot corpus shortfall; no duplicate padding or fallback')
    # Production exactness independently cross-checks every selected generic source.
    admitted=0
    with SynthesisClient(digest(PROFILE),limits=limits,cache_root=Path(output)/'private/admission-cache') as client:
        for rs in selected['splits'].values():
            for r in rs:
                if time.monotonic()>=deadline:raise TimeoutError('complete native admission deadline')
                result=client.execute(r['source'],r['indices'],deadline=min(deadline,time.monotonic()+30))
                if result.get('outcome')!='ok' or result.get('values')!=r['values']:
                    raise ValueError('selected generic source lacks complete independent production agreement')
                admitted+=1
                if admitted%256==0:replace_json(Path(output)/'progress.json',dict(admitted=admitted,seconds=time.monotonic()-begin))
    m=publish(selected,prompts,truth,root,source=config['source'],recipe=recipe(),sampled=counter+1,exclusions=excluded)
    check(root,hidden=True);write(Path(output)/'corpus-audit.json',m['audit'])
    return dict(status='COMPLETE',work=m['identity'],prepared='private/prepared',counts=m['audit']['counts'],
        native_admitted=admitted,seconds=time.monotonic()-begin)
