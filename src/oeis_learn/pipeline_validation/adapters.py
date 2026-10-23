"""Reuse the existing installed-source, role isolation and qualification owners."""
import copy
import json
from pathlib import Path
import sys

from .engines import Images
from .identity import write
from .resources import observe, resolve
from .runner import CapabilityError
from .supervision import launch


class Adapters:
    def __init__(self, args, root, host_factory, contract):
        self.args=args;self.root=Path(root);self.host_factory=host_factory;self.contract=contract
        self.hosts={};self.images=Images(contract['budgets']['build_per_engine_seconds'])

    def host(self,name):
        if name not in self.hosts:
            from oeis_learn.container_runtime import Engine
            options=copy.copy(self.args);options.engine=name
            try:
                engine=Engine(name);facts=engine.facts()
                measured=observe();measured['limits'].append(facts['memory_bytes'])
                policy=resolve(**measured,profile=self.contract['resource_policy'],lower=int(options.memory_gib*(1<<30)) if options.memory_gib is not None else None)
                options.memory_gib=policy['ceiling_bytes']/(1<<30)
                options.cpus=min(options.cpus or 16,policy['cpus'])
                options.pipeline_resource_policy=policy
                host=self.host_factory(options)
                host.policy['pipeline_resources']=policy
                self.hosts[name]=host
            except (RuntimeError,OSError,ValueError) as exc:
                raise CapabilityError(f'{name}: {exc}') from exc
        return self.hosts[name]

    def __call__(self,stage,output):
        if stage.adapter=='host':
            # Run only stdlib cases here; native/Torch suites belong inside the image.
            names=['contracts','resources','bundle','reuse','validation']
            return launch([sys.executable,'-S','-m','unittest',*[f'tests.unit.test_pipeline_{n}' for n in names]],output,seconds=stage.seconds)
        parts=stage.id.split('.')
        name=parts[1] if parts[0] in {'engine','image','cpu','operations'} else self.args.engine
        host=self.host(name)
        if stage.adapter=='engine':
            facts=host.engine.facts();write(output/'engine.json',dict(facts=facts,resources=host.policy['pipeline_resources']))
            return facts
        if stage.adapter=='image':
            target=parts[2]
            image=self.images.ensure(host,target,output/'build')
            # Whole installed package inspection is mandatory for pinned and built images.
            from oeis_learn.container_validation import inspect_current_image
            inspect_current_image(host,image,output/'inspection',seconds=stage.seconds)
            write(output/'image.json',dict(image=image,target=target,engine=name))
            return dict(image=image)
        target='gpu' if stage.adapter=='hip' else 'qualification'
        image=self.images.images[(name,target)]
        host.image=image
        if stage.adapter=='hip':
            from oeis_learn.container_validation import _stages
            host.select_gpu()
            with host.gpu_lock():
                return _stages(host,'bigint-gpu',output)[1]
        from oeis_learn.container_validation import _stages
        return _stages(host,'bigint-retirement' if stage.adapter=='retirement' else 'bigint-cpu',output)[1]
