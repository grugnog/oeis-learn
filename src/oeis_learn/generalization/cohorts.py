"""Bounded deterministic marginal quota selection and least-privilege projection."""
from dataclasses import dataclass
import json
from .contracts import shape

@dataclass(frozen=True)
class CohortChoice:
    payload: bytes
    shortfalls: tuple[str,...]

    @property
    def rows(self): return tuple(json.loads(self.payload))
    @property
    def complete(self): return not self.shortfalls


def choose(rows, *, count=128, quotas, structural=False, draw_cap=1_000_000):
    if type(count) is not int or not 1<=count<=32768 or not 1<=draw_cap<=1_000_000 or any(type(v) is not int or not 0<=v<=count for v in quotas.values()):
        raise ValueError('invalid frozen quota')
    candidates=[]; seen=set()
    for i,row in enumerate(rows):
        if i>=draw_cap: break
        if row['id'] in seen: raise ValueError('duplicate cohort row')
        seen.add(row['id']); candidates.append(row)
    candidates.sort(key=lambda r:r['id'])
    chosen=[]; counts={k:0 for k in quotas}; families={k:set() for k in ('loop_active','branch_active')}
    # Gains only decrease as quotas/family minima fill. Lazy upper bounds avoid
    # rescanning a 32,768-row shard for every choice, preserving the original
    # maximum-gain / smallest-id ordering exactly.
    import heapq
    def gain(r):
        return sum(k in r['tags'] and counts[k]<v for k,v in quotas.items())+sum(structural and k in r['tags'] and len(fs)<2 and r['family'] not in fs for k,fs in families.items())
    heap=[(-gain(r),r['id'],i) for i,r in enumerate(candidates)]
    heapq.heapify(heap)
    while heap and len(chosen)<count:
        old,key,i=heapq.heappop(heap);row=candidates[i]
        current=-gain(row)
        if current!=old:
            heapq.heappush(heap,(current,key,i));continue
        chosen.append(row)
        for k in counts:counts[k]+=k in row['tags']
        for k in families:
            if k in row['tags']:families[k].add(row['family'])
    shortfalls=[]
    if len(chosen)<count: shortfalls.append(f'rows:{count-len(chosen)}')
    shortfalls.extend(f'{k}:{v-counts[k]}' for k,v in sorted(quotas.items()) if counts[k]<v)
    if structural: shortfalls.extend(f'{k}_families:{2-len(fs)}' for k,fs in families.items() if len(fs)<2)
    return CohortChoice(json.dumps(chosen,sort_keys=True).encode(),tuple(shortfalls))


def project(row):
    if row['role'] not in {'training','selection','assessment'} or len(row['values'])!=100 or len(row['indices'])!=100:
        raise ValueError('invalid role/horizon')
    visible={k:row[k] for k in ('id',)}
    visible.update(indices=list(row['indices'][:20]),values=list(row['values'][:20]))
    shape('visible-prompt',visible)
    # Scorer/training payload stays in its own role mount. Opaque ids must already
    # have been assigned by preparation, never encode protected family labels.
    protected={k:row[k] for k in ('id','role','indices','values','source')}
    if row['role']=='training':
        from oeis_learn.decoder.bigint_codec import encode_body
        protected['tokens']=encode_body(row['source'])
        shape('training-target',protected)
    else:
        if row.get('holdout') not in {'parameter','structural'}:raise ValueError('registered protected stratum required')
        protected.update(family=row['family'],stratum=row['holdout'])
        shape('scorer-target',protected)
    return visible,protected


def publish(choices, preparation, proposed, manifests, *, root):
    from .contracts import resolve_study
    shortfalls=tuple(f'{cell}:{reason}' for cell,choice in choices.items() for reason in choice.shortfalls)
    if not choices: shortfalls=('missing cells',)
    return resolve_study(preparation,proposed,manifests,root=root,shortfalls=shortfalls)


def pac_inventory(rows,probe_draws,*,pool_identity,draw_identity,seed,batch_size,final_collision_ids,exclusions):
    """Admit finite, already executed metadata; no draws, reseeding or row removal.

    C owns source/output hash and native-receipt verification. Hash-order anchor
    selection is outcome blind and retains every original gradient row/offset.
    """
    from collections import Counter,defaultdict
    from .contracts import pac_record
    from .exclusions import pac_probe_exclusions
    from .budgets import integer
    from oeis_learn.pipeline_validation.identity import digest
    integer(seed);integer(batch_size,1)
    row_fields={'id','family','offset','source_hash','output_hash','window_hash','target_stratum','length_stratum','role'}
    probe_fields={'id','family','draw_index','source_hash','output_hash','window_hash','native_hash','role'}
    if (len(rows)!=32768 or digest(rows)!=pool_identity or digest(probe_draws)!=draw_identity
        or any(set(r)!=row_fields or r['role']!='training' for r in rows)
        or any(set(r)!=probe_fields or r['role']!='training' for r in probe_draws)):
        raise ValueError('original finite training/probe identity or role mismatch')
    counts=Counter(r['family'] for r in rows)
    if len(counts)!=32 or min(counts.values())<max(4,batch_size):raise ValueError('32 nonempty families with distinct batch/anchor capacity required')
    if len({r['id'] for r in rows})!=32768 or [r['offset'] for r in rows]!=list(range(32768)):
        raise ValueError('original row identity/order/offset mismatch')
    if len({r['id'] for r in probe_draws})!=len(probe_draws) or {r['id'] for r in rows}&{r['id'] for r in probe_draws}:
        raise ValueError('distinct probe identities required')
    grouped=defaultdict(list)
    for r in probe_draws:
        if r['family'] not in counts or type(r['draw_index']) is not int or not 0<=r['draw_index']<256:
            raise ValueError('frozen finite probe draw bound')
        if r['native_hash']!=r['output_hash']:raise ValueError('probe native disagreement')
        grouped[r['family']].append(r)
    for family in counts:
        draws=grouped[family]
        if not 8<=len(draws)<=256 or [r['draw_index'] for r in draws]!=list(range(len(draws))):
            raise ValueError('finite contiguous frozen probe reservoir required')
    bad,shared=pac_probe_exclusions(rows,probe_draws,final_collision_ids=final_collision_ids)
    families=[]
    for family in sorted(counts):
        probes=[r['id'] for r in grouped[family] if r['id'] not in bad][:8]
        if len(probes)!=8:raise ValueError('probe quota deficit; no reseed or relaxation')
        candidates=sorted((r for r in rows if r['family']==family),key=lambda r:digest(['pac-anchor-order/v1',seed,family,r['id']]))
        strata={};anchors=[]
        for r in candidates:strata.setdefault((r['target_stratum'],r['length_stratum']),r)
        for _,r in sorted(strata.items(),key=lambda kv:digest(['pac-anchor-stratum/v1',seed,family,kv[0]])):
            if len(anchors)<4:anchors.append(r['id'])
        for r in candidates:
            if len(anchors)==4:break
            if r['id'] not in anchors:anchors.append(r['id'])
        expected={(t,l) for t in ('supported','fallback') for l in ('short','long')}
        families.append(dict(id=family,rows=[r['id'] for r in rows if r['family']==family],probes=probes,anchors=anchors,
                             missing_strata=['/'.join(s) for s in sorted(expected-set(strata))]))
    retained=[{k:v for k,v in r.items() if k not in {'role','window_hash'}} for r in rows]
    record=dict(version='pac-inventory/v1',role='training',families=families,rows=retained,sidecar=pool_identity,
                probe_seed=seed,probe_draws=[dict(family=f,draws=len(grouped[f])) for f in sorted(counts)],
                source_dispositions=[dict(reason='original_retained',count=len(rows))],
                exclusion_dispositions=[dict(reason=k,count=v) for k,v in sorted(Counter(x for reasons in bad.values() for x in reasons).items())],
                shared_training_windows=shared,exclusions=exclusions)
    return pac_record('inventory',record,sealed=False).record()
