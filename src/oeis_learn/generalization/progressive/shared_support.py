"""Immutable in-memory support and compact calibration transport.

Canonical identities remain byte-for-byte equivalent to the original JSON
records. Only the transport is factored; installed consumers reconstruct and
verify the complete original package before validating or using it.
"""
import hashlib
import json
from .contracts import digest, seal, verify_refs, atomic_write, reference


def _blocked(*args, **kwargs):
    raise TypeError('shared support is immutable')


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _blocked
    __setattr__ = __delattr__ = _blocked
    def __deepcopy__(self, memo):return self


class FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _blocked
    __setattr__ = __delattr__ = _blocked
    def __deepcopy__(self, memo):return self


def freeze(value):
    if isinstance(value,(FrozenDict,FrozenList)):return value
    if isinstance(value,dict):return FrozenDict((k,freeze(v)) for k,v in value.items())
    if isinstance(value,list):return FrozenList(map(freeze,value))
    return value


def raw(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def shared(package):
    """Detach once from mutable input; retain only one canonical support buffer."""
    if isinstance(package.get('shards'),FrozenList):return package
    if seal(package)!=package:raise ValueError('original sealed support package required')
    value=dict(package)
    value['shards']=freeze(package['shards'])
    object.__setattr__(value['shards'],'_canonical',raw(value['shards']))
    return value


def package_digest(value):
    # The costly immutable tree has one canonical byte representation. Other
    # fields are small, independently serialized and never trusted from a cache.
    h=hashlib.sha256();h.update(b'{')
    for i,key in enumerate(sorted(value)):
        if i:h.update(b',')
        h.update(raw(key));h.update(b':')
        item=value[key]
        h.update(item._canonical if isinstance(item,FrozenList) and hasattr(item,'_canonical') else raw(item))
    h.update(b'}')
    return 'sha256:'+h.hexdigest()


def seal_package(value):
    payload={k:v for k,v in value.items() if k!='identity'}
    return dict(payload,identity=package_digest(payload))


def validated_rows(shards, *, native_verified, short=False):
    """Validate immutable support once per scope; mutable inputs never use a cache.

    The cache belongs to the detached frozen tree, not an identity supplied by a
    caller or a process-global table. Task/profile/supply checks remain separate.
    """
    from .training_rows import validate_row
    key=(native_verified,short)
    cacheable=isinstance(shards,FrozenList) and hasattr(shards,'_canonical')
    cached=getattr(shards,'_validated_rows',{}) if cacheable else {}
    if key in cached:return cached[key]
    seen={}
    for shard in shards:
        if not isinstance(shard,list) or not 1<=len(shard)<=4096:
            raise ValueError('bounded nonempty verified shard required')
        for row in shard:
            validate_row(row,native_verified=native_verified,short=short)
            if row['draw_identity'] in seen and seen[row['draw_identity']]!=row:
                raise ValueError('conflicting training row')
            seen[row['draw_identity']]=row
    if cacheable:
        seen=FrozenDict(seen)
        object.__setattr__(shards,'_validated_rows',FrozenDict({**cached,key:seen}))
    return seen


def projection_size(cases):
    """Exact pretty-JSON byte counts plus atomic-publication headroom."""
    encoder=json.JSONEncoder(sort_keys=True,indent=2,allow_nan=False)
    def size(value):return 1+sum(len(part.encode()) for part in encoder.iterencode(value))
    bases={};total=0
    for case in cases:
        p=case['package'];shards=p['shards'];key=id(shards)
        if key not in bases:bases[key]=size(shards)
        # Fixed-length sha256/path fields, including the wrapper's own seal.
        ref=dict(path='inputs/support/'+('0'*64)+'.json',bytes=bases[key],sha256='sha256:'+('0'*64))
        envelope=seal(dict(version='progressive-calibration-input/v1',support=ref,
            package={k:v for k,v in p.items() if k!='shards'}))
        total+=size(envelope)
    return 2*(sum(bases.values())+total)


def project(root,cases):
    from pathlib import Path
    root=Path(root);bases={};result=[]
    for case in cases:
        p=case['package'];key=id(p['shards'])
        if key not in bases:
            identity=digest(p['shards'])
            path=root/'inputs'/'support'/(identity[7:]+'.json')
            if not path.exists():atomic_write(path,p['shards'])
            bases[key]=reference(root,path)
        envelope=seal(dict(version='progressive-calibration-input/v1',
            support=bases[key],package={k:v for k,v in p.items() if k!='shards'}))
        path=root/'inputs'/(case['compatibility'][7:]+'.json')
        atomic_write(path,envelope)
        result.append(dict(compatibility=case['compatibility'],artifact=reference(root,path)))
    return result


def restore(root,ref):
    value=json.loads(verify_refs(root,[ref],max_bytes=256<<20)[0])
    if value.get('version')=='progressive-calibration-input/v1':
        if set(value)!={'version','identity','support','package'} or seal(value)!=value:
            raise ValueError('closed compact calibration input required')
        if 'shards' in value['package']:raise ValueError('ambiguous shared calibration support')
        shards=json.loads(verify_refs(root,[value['support']],max_bytes=256<<20)[0])
        value=dict(value['package'],shards=shards)
    if seal(value)!=value:raise ValueError('reconstructed calibration identity differs')
    return value
