"""Owned bounded reference batches; cache reuse never qualifies native execution."""
import gzip
import json
from pathlib import Path
import shutil
from .identity import seal,digest,file_hash,confined,validate
from oeis_learn.fresh.common import write as atomic_write

MAX_BYTES=1<<30
MAX_BATCH_BYTES=16<<20


def binding(sampler,limits):
    from oeis_learn.synthesis.admission import reference_identity
    from oeis_learn.data import bigint_programs
    return digest(dict(version='pipeline-pilot-reference-semantics/v1',sampler=sampler.identity,
        resources=limits.identity,reference=reference_identity(),generator=file_hash(Path(bigint_programs.__file__)),
        serialization=file_hash(Path(__file__)),horizon=list(range(100))))


class ReferenceCache:
    @staticmethod
    def index(root):
        path=confined(root,'index.json')
        if path.stat().st_size>1<<20:raise ValueError('reference cache index exceeds bound')
        return json.loads(path.read_text())

    def __init__(self,root,expected,*,seed=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.expected=expected;self.pending=[]
        self.path=self.root/'index.json'
        if seed is not None:
            if self.path.exists():raise ValueError('cannot replace a reference cache prefix')
            source=Path(seed);value=self.index(source)
            self._check(value)
            for item in value['batches']:
                path=confined(source,item['path'])
                if file_hash(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('reference cache batch changed')
                shutil.copyfile(path,self.root/item['path'])
            atomic_write(self.path,value)
        if self.path.exists():
            self.value=self.index(self.root);self._check(self.value)
        else:
            self.value=seal(dict(version='pipeline-pilot-reference-cache/v1',binding=expected,count=0,bytes=0,batches=[]))
            atomic_write(self.path,self.value)

    def _check(self,value):
        from oeis_learn.synthesis.state import require_hash
        validate(value,{'version','binding','count','bytes','batches'},'pipeline-pilot-reference-cache/v1')
        require_hash(value['binding'])
        if value['binding']!=self.expected:raise ValueError('reference semantics changed; cache cannot be reused')
        if (type(value['count']) is not int or not 0<=value['count']<=262144 or type(value['bytes']) is not int
            or not 0<=value['bytes']<=MAX_BYTES or not isinstance(value['batches'],list) or len(value['batches'])>2048):
            raise ValueError('finite reference cache bound exceeded')
        count=size=0
        for batch in value['batches']:
            if (set(batch)!={'path','sha256','bytes','first','count'} or batch['first']!=count
                or type(batch['count']) is not int or not 1<=batch['count']<=256 or type(batch['bytes']) is not int
                or not 0<batch['bytes']<=MAX_BYTES):
                raise ValueError('reference cache prefix has a gap/duplicate')
            require_hash(batch['sha256'])
            if batch['path']!=f'batch-{count:06d}.json.gz':raise ValueError('closed reference cache member required')
            count+=batch['count'];size+=batch['bytes']
        if count!=value['count'] or size!=value['bytes']:raise ValueError('reference cache counters changed')

    @property
    def count(self):return self.value['count']

    def records(self):
        for item in self.value['batches']:
            path=confined(self.root,item['path'])
            if file_hash(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('reference cache batch changed')
            with gzip.open(path,'rb') as stream:raw=stream.read(MAX_BATCH_BYTES+1)
            if len(raw)>MAX_BATCH_BYTES:raise ValueError('reference cache decoded batch exceeds bound')
            rows=json.loads(raw)
            if len(rows)!=item['count']:raise ValueError('reference batch count changed')
            for i,row in enumerate(rows):
                if set(row)!={'counter','row','exclusion'} or row['counter']!=item['first']+i:
                    raise ValueError('reference cache counter changed')
                yield row

    def append(self,counter,row,exclusion=None):
        if counter!=self.count+len(self.pending) or counter>=262144:raise ValueError('reference cache must append one finite original counter')
        if (row is None and (not isinstance(exclusion,str) or not exclusion)) or (row is not None and (not isinstance(row,dict) or exclusion is not None)):
            raise ValueError('reference row or explicit exclusion required')
        self.pending.append(dict(counter=counter,row=row,exclusion=exclusion))
        if len(self.pending)==256:self.flush()

    def flush(self):
        if not self.pending:return
        raw=json.dumps(self.pending,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>MAX_BATCH_BYTES:raise ValueError('reference cache batch exceeds byte bound')
        compressed=gzip.compress(raw,compresslevel=3,mtime=0)
        if self.value['bytes']+len(compressed)>MAX_BYTES:raise ValueError('reference cache exceeds1GiB')
        first=self.count;name=f'batch-{first:06d}.json.gz';path=self.root/name
        with path.open('xb') as stream:stream.write(compressed)
        self.value['batches'].append(dict(path=name,sha256=file_hash(path),bytes=len(compressed),first=first,count=len(self.pending)))
        self.value['count']+=len(self.pending);self.value['bytes']+=len(compressed)
        self.value=seal(self.value);atomic_write(self.path,self.value);self.pending=[]
