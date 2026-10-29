"""One image ensure per engine/target with shared finite build allowance."""
import json
import time
from .supervision import launch


class Images:
    def __init__(self, cap=14400):
        self.cap=cap;self.used={};self.images={}

    def ensure(self, host, target, output, *, seconds=None):
        key=(host.engine.name,target)
        if key in self.images:
            return self.images[key]
        remaining=min(self.cap-self.used.get(key[0],0), seconds if seconds is not None else self.cap)
        if remaining<=0:
            raise TimeoutError('shared engine build allowance exhausted')
        start=time.monotonic()
        try:
            if target=='gpu' and host.args.image:
                image=host.engine.image(host.args.image)
            else:
                tag='oeis-learn:pipeline-'+target
                argv=[host.engine.executable,'build','--target',target,'--tag',tag]
                if host.args.gpu_base_image:
                    argv+=['--build-arg','GPU_BASE_IMAGE='+host.args.gpu_base_image]
                launch([*argv,host.policy['workspace']],output,seconds=remaining,stream_cap=host.policy.get('pipeline_budgets',{}).get('stream_bytes',8<<20))
                image=host.engine.image(tag)
            self.images[key]=image
            return image
        finally:
            self.used[key[0]]=self.used.get(key[0],0)+time.monotonic()-start


def junit(path):
    import xml.etree.ElementTree as ET
    cases=list(ET.parse(path).getroot().iter('testcase'))
    counts={name:sum(c.find(name) is not None for c in cases) for name in ('skipped','failure','error')}
    if not cases or any(counts.values()):
        raise ValueError('mandatory JUnit empty/skipped/failed: '+str(path))
    return dict(tests=len(cases),**counts)
