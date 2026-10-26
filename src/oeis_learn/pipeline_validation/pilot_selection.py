"""Select a disjoint actual corpus; unused bridge candidates are not examples."""
from collections import Counter,defaultdict
from functools import lru_cache
import time
from oeis_learn.synthesis.contracts import identity,canonical_integer

POLICY='selected-disjoint-windows/v1'


def window_keys(row):
    values=row['values'];indices=row['indices']
    if len(values)!=100 or len(indices)!=100:raise ValueError('complete100-index candidate required')
    ns=[canonical_integer(n) for n in indices]
    for v in values:canonical_integer(v,bits=16384)
    if any(a>=b for a,b in zip(ns,ns[1:])):raise ValueError('increasing candidate indices required')
    for i in range(81):yield identity([values[i:i+20],[n-ns[i] for n in ns[i:i+20]]])


class Windows:
    def __init__(self,blocked=()):
        self.used=set()
        for row in blocked:self.used.update(window_keys(row))

    def candidate(self,row,local=()):
        keys=set()
        for key in window_keys(row):
            if key in self.used or key in local:return None
            keys.add(key)
        return keys

    def admit(self,row):
        keys=self.candidate(row)
        if keys is None:return False
        self.used.update(keys);return True


class Shortfall(ValueError):
    def __init__(self,audit):
        self.audit=audit
        super().__init__('pilot corpus selection shortfall: '+str(audit['counts'])+'; required '+str(audit['required']))


def select(rows,*,training,parameter,structural,blocked=(),deadline=None,progress=None):
    """Production fixes counts in its wrapper; small counts support counterexamples."""
    from oeis_learn.fresh.bigint_splits import behavior
    from oeis_learn.decoder.bigint_codec import encode_body
    if min(training,parameter,structural)<=0 or training<4*parameter:raise ValueError('matched four-plus-one selection required')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate candidate identity')
    guard=Windows(blocked);families=defaultdict(list)
    for r in rows:families[r['family']].append(r)
    keys=sorted((k for k,rs in families.items() if len({r['variant'] for r in rs})>=16),
        key=lambda k:identity(['structural-reservation/v1',k]))
    key_set=set(keys)
    reserved=set(keys[::4]);splits=dict(training=[],parameter_development=[],structural_development=[])
    limits=dict(training=training,parameter_development=parameter,structural_development=structural)
    rejected=Counter();facts={};paired={};family_ok={};selected_ids=set();counts={k:[0,0] for k in splits}

    last_report=0.;phase='structural'

    def audit():
        return dict(policy=POLICY,phase=phase,counts={k:len(v) for k,v in splits.items()},required=limits,
            candidate_rows=len(rows),candidate_families=len(keys),qualified_families=sum(family_ok.values()),
            stages={k:dict(Counter(r['stage'] for r in rs)) for k,rs in splits.items()},
            selected_families={k:len({r['family'] for r in rs}) for k,rs in splits.items()},
            behavior_counts={k:dict(constant=v[0],offset_only=v[1]) for k,v in counts.items()},
            rejections=dict(rejected),unused_candidates_are_not_group_bridges=True)

    def tick(force=False):
        nonlocal last_report
        now=time.monotonic()
        expired=deadline is not None and now>=deadline
        if progress is not None and (force or expired or now-last_report>=10):
            progress(audit());last_report=now
        if expired:raise TimeoutError('finite pilot corpus selection deadline')

    tick(True)

    @lru_cache(maxsize=32768)
    def encoded(source):return tuple(encode_body(source))

    def tokens(row):
        if row['id'] not in paired:
            paired[row['id']]=dict(concrete=encoded(row['source']),parameterized=encoded(row['parameter_source']))
        return paired[row['id']]

    def valid_family(key):
        if key not in family_ok:
            variants=set()
            for row in sorted(families[key],key=lambda r:r['id']):
                try:tokens(row);variants.add(row['variant'])
                except ValueError:rejected['codec']+=1
                if len(variants)==16:break
            family_ok[key]=len(variants)==16
        return family_ok[key]

    def eligible(row,split,local=(),extra=(0,0)):
        tick()
        if row['id'] in selected_ids:return None
        if not valid_family(row['family']):return None
        try:tokens(row)
        except ValueError:rejected['codec']+=1;return None
        if row['id'] not in facts:facts[row['id']]=behavior(row)
        c,o,_=facts[row['id']]
        if any(counts[split][i]+extra[i]+v>limits[split]//10 for i,v in enumerate((c,o))):
            rejected['behavior_cap']+=1;return None
        windows=guard.candidate(row,local)
        if windows is None:rejected['overlap']+=1
        return windows

    def commit(row,split,windows):
        splits[split].append(row);selected_ids.add(row['id']);guard.used.update(windows)
        for i,v in enumerate(facts[row['id']][:2]):counts[split][i]+=v

    # Deterministic structural reserve, then one matched parameter case per family.
    structural_rows=[r for k in keys if k in reserved for r in families[k]]
    structural_rows.sort(key=lambda r:(not behavior(r)[2],r.get('stage',''),r['id']))
    for row in structural_rows:
        windows=eligible(row,'structural_development')
        if windows is not None:commit(row,'structural_development',windows)
        if len(splits['structural_development'])==structural:break
    phase='matched-parameter';tick(True)
    for key in keys:
        if key in reserved:continue
        if len(splits['parameter_development'])==parameter:break
        batch=[];local=set();extra=[0,0]
        for row in sorted(families[key],key=lambda r:r['id']):
            split='training' if len(batch)<4 else 'parameter_development'
            windows=eligible(row,split,local,extra if len(batch)<4 else (0,0))
            if windows is None:continue
            batch.append((row,split,windows));local.update(windows)
            if len(batch)<=4:
                for i,v in enumerate(facts[row['id']][:2]):extra[i]+=v
            if len(batch)==5:break
        if len(batch)==5:
            for row,split,windows in batch:commit(row,split,windows)
    phase='training';tick(True)
    pools=defaultdict(list)
    for row in sorted(rows,key=lambda r:r['id']):
        if row['family'] in key_set and row['family'] not in reserved:pools[row['stage']].append(row)
    positions={k:0 for k in pools}
    while len(splits['training'])<training:
        advanced=False
        for stage in sorted(pools):
            while positions[stage]<len(pools[stage]):
                row=pools[stage][positions[stage]];positions[stage]+=1
                windows=eligible(row,'training')
                if windows is None:continue
                commit(row,'training',windows);advanced=True;break
            if len(splits['training'])==training:break
        if not advanced:break
    phase='complete';tick(True);result_audit=audit()
    if result_audit['counts']!=limits:raise Shortfall(result_audit)
    return dict(splits=splits,selection_audit=result_audit,
        paired={r['id']:paired[r['id']] for rs in splits.values() for r in rs})
