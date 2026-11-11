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


def projection_size(cases):
    """Conservative compact projection bound, before any per-case publication."""
    bases={};total=0
    for case in cases:
        p=case['package'];shards=p['shards']
        key=id(shards)
        if key not in bases:bases[key]=len(raw(shards))
        total+=len(raw({k:v for k,v in p.items() if k!='shards'}))+4096
    # atomic_write uses indentation; conservative 4x compact bytes covers it,
    # plus separate original package metadata and every content reference.
    return 4*(sum(bases.values())+total)


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
