"""Bounded per-replay immutable input cache; no device measurements change."""
import json

from oeis_learn.generalization.progressive.contracts import seal, verify_refs
from oeis_learn.generalization.progressive.shared_support import FrozenList, freeze, raw, package_digest


class Inputs:
    def __init__(self, root):
        self.root = root
        self.support = {}
        self.bytes = 0

    def restore(self, ref):
        value = json.loads(verify_refs(self.root, [ref], max_bytes=256 << 20)[0])
        if value.get('version') == 'progressive-calibration-input/v1':
            if set(value) != {'version', 'identity', 'support', 'package'} or seal(value) != value or 'shards' in value['package']:
                raise ValueError('closed original compact calibration input required')
            support = value['support']; key = (support['path'], support['bytes'], support['sha256'])
            if key not in self.support:
                if 16*(self.bytes+support['bytes'])+(512<<20)>16<<30:
                    raise ValueError('bounded calibration replay memory inventory required')
                rows = freeze(json.loads(verify_refs(self.root, [support], max_bytes=256<<20)[0]))
                if not isinstance(rows, FrozenList):
                    raise ValueError('original calibration shards required')
                object.__setattr__(rows, '_canonical', raw(rows))
                self.support[key] = rows; self.bytes += support['bytes']
            value = dict(value['package'], shards=self.support[key])
        if value.get('identity') != package_digest({k:v for k,v in value.items() if k!='identity'}):
            raise ValueError('reconstructed original calibration input identity differs')
        return value


def maximum_target(package):
    rows = package['shards']
    cached = getattr(rows, '_maximum_target_tokens', None) if isinstance(rows, FrozenList) else None
    if cached is None:
        cached = max(len(r['tokens']) for shard in rows for r in shard)
        if isinstance(rows, FrozenList): object.__setattr__(rows, '_maximum_target_tokens', cached)
    return cached
