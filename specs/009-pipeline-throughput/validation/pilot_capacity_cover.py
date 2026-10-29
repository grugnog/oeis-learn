"""Independent sparse-window clique-cover ceiling for this fixed candidate pool."""
import sys,json,gzip,time,heapq,random,itertools,argparse,re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from oeis_learn.synthesis.contracts import identity
from oeis_learn.data.bigint_programs import BigintSampler

def cover(groups,n):
 active=bytearray([1])*n;remaining=n;heap=[(-len(ids),key) for key,ids in groups.items()];heapq.heapify(heap);chosen=[]
 while remaining:
  old,key=heapq.heappop(heap);ids=groups[key];current=sum(active[i] for i in ids)
  if not current:continue
  if current!=-old:heapq.heappush(heap,(-current,key));continue
  chosen.append(key)
  for i in ids:
   if active[i]:active[i]=0;remaining-=1
 return chosen
# Check the ceiling against exact independent-set enumeration on small systems.
rng=random.Random(20261016)
for trial in range(64):
 rows=[set(rng.sample(range(8),rng.randint(1,4))) for _ in range(8)];groups=defaultdict(list)
 for i,keys in enumerate(rows):
  for k in keys:groups[k].append(i)
 chosen=cover(groups,8)
 assert set().union(*(set(groups[k]) for k in chosen))==set(range(8))
 optimum=0
 for mask in range(256):
  subset=[i for i in range(8) if mask>>i&1]
  if len(subset)>optimum and all(not rows[a]&rows[b] for a,b in itertools.combinations(subset,2)):optimum=len(subset)
 assert len(chosen)>=optimum
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch-dir',type=Path,required=True)
parser.add_argument('--checkpoint',type=int,default=262144,choices=(65536,131072,262144,524288))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();limit=args.checkpoint;begin=time.monotonic()
paths=sorted(args.batch_dir.glob('*.json.gz'),key=lambda p:int(re.search(r'\d+',p.name).group()))
def rows():
    count=0
    for p in paths:
        offset=int(re.search(r'\d+',p.name).group())
        for j,item in enumerate(json.load(gzip.open(p,'rt'))):
            counter=item['counter'] if 'counter' in item and 'row' in item else offset+j
            if counter>=limit:break
            if counter!=count:raise ValueError('contiguous original-counter prefix required')
            count+=1;row=item['row'] if 'counter' in item and 'row' in item else item
            if row is not None:yield row
        if count==limit:break
    if count!=limit:raise ValueError('complete requested reference prefix required')
variants=defaultdict(set)
for row in rows():variants[row['family']].add(row['variant'])
keys=sorted((k for k,v in variants.items() if len(v)>=16),key=lambda k:identity(['structural-reservation/v1',k]));reserved=set(keys[::4]);eligible=set(keys)-reserved
print('FAMILIES',len(keys),round(time.monotonic()-begin,2),flush=True)
offsets=[0,1,2,3,4,5,10,20,30,40,50,60,70,75,76,77,78,79,80];groups=defaultdict(list);seen={};raw=0;ns=list(map(str,range(100)))
for row in rows():
    if row['family'] not in eligible:continue
    raw+=1
    if row['indices']!=ns or len(row['values'])!=100:raise ValueError('fixed complete generic100-index reference required')
    signature=identity(row['values'])
    if signature in seen:continue
    i=len(seen);seen[signature]=i
    windows={identity([row['values'][j:j+20],list(range(20))]) for j in offsets}
    for key in windows:groups[key].append(i)
print('COVERING',len(seen),len(groups),round(time.monotonic()-begin,2),flush=True)
chosen=cover(groups,len(seen))
r=dict(version='pilot-sparse-window-clique-cover/v1',counters=limit,seed=20261016,candidate_families=len(keys),reserved_families=len(reserved),nonreserved_rows=raw,distinct_complete_horizons=len(seen),window_offsets=offsets,window_classes=len(groups),training_ceiling=len(chosen),required_training=32768,seconds=time.monotonic()-begin,small_exact_system_checks=64,
 reasoning='All nonreserved candidates are covered by chosen actual20-term-window cliques. Any pairwise-window-disjoint training set can contain at most one row per clique, so it cannot exceed this cover size. Collapsing identical100-term rows is exact. Other windows, codec/behavior and parameter/OEIS/structural exclusions can only reduce the true maximum. This is a fixed data-pool ceiling, not a learned model capacity or runtime qualification.',native_or_hip_qualification=False,cover_identity=identity(sorted(chosen)))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
