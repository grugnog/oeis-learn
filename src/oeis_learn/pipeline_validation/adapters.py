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
        self.receipts={}
        from .benchmarks import Benchmarks
        self.benchmarks=Benchmarks(self)
        from .replay import ReplayStages
        self.replay=ReplayStages(self)
        from .replay_comparison import ComparisonStages
        self.replay_comparison=ComparisonStages(self.replay)
        from .replay_batching import BatchingStages
        self.replay_batching=BatchingStages(self)
        from .hardware_ladder import HardwareStages
        self.hardware=HardwareStages(self)
        from .native_capacity import NativeCapacityStages
        self.native_capacity=NativeCapacityStages(self)
        from .coschedule import CoscheduleStages
        self.coschedule=CoscheduleStages(self)
        from .model_sizing import SizingStages
        self.sizing=SizingStages(self)
        from .soak import SoakStages
        self.soak=SoakStages(self)
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
                host.policy['pipeline_budgets']=self.contract['budgets']
                host.policy['cpu_memory_bytes']=policy['ceiling_bytes']
                self.hosts[name]=host
            except (RuntimeError,OSError,ValueError) as exc:
                raise CapabilityError(f'{name}: {exc}') from exc
        return self.hosts[name]

    def stage_binding(self,stage,base):
        if stage.adapter!='cpu':return base
        import os
        from .identity import seal,digest
        name=stage.id.split('.')[1];host=self.host(name)
        policy={k:v for k,v in host.policy['pipeline_resources'].items() if k not in {'identity','available_bytes'}}
        try:
            cpu='\n'.join(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                if not line.startswith(('cpu MHz','bogomips')))
        except OSError:cpu='unavailable'
        hardware=dict(system=list(os.uname()),cpu_inventory=digest(cpu))
        return dict(base,stage_version='pipeline-pure-tests/v1',image=self.images.images[(name,'qualification')],
            engine=host.policy['runtime'],security=host.policy['engine_info'],resources=seal(policy),hardware=hardware)

    def __call__(self,stage,output):
        if stage.adapter=='host':
            # Run only stdlib cases here; native/Torch suites belong inside the image.
            names=['contracts','resources','bundle','reuse','validation','comparisons','policy','native_pool','cache','overlap','overlap_metrics','qualification','studies','supervision','benchmarks','memory','disk','native_measurements','attempts','study_batch','workloads','role_failures','replay','replay_comparison','replay_batching','hardware_ladder','native_capacity','coschedule','model_sizing','soak']
            code = 'import sys,unittest;sys.path[:0]='+repr([str(self.root/'src'),str(self.root)])+';unittest.main(module=None,argv='+repr(['unittest',*[f'tests.unit.test_pipeline_{n}' for n in names]])+')'
            return launch([sys.executable,'-S','-c',code],output,seconds=stage.seconds,stream_cap=self.contract['budgets']['stream_bytes'])
        if stage.adapter.startswith('sizing-'):
            return self.sizing(stage,output)
        if stage.adapter.startswith('soak-'):
            return self.soak(stage,output)
        if stage.adapter.startswith('coschedule-'):
            return self.coschedule(stage,output)
        if stage.adapter.startswith('native-capacity'):
            return self.native_capacity(stage,output)
        if stage.adapter.startswith('hardware-'):
            return self.hardware(stage,output)
        if stage.adapter.startswith('replay-batching-'):
            return self.replay_batching(stage,output)
        if stage.adapter.startswith('replay-comparison-'):
            return self.replay_comparison(stage,output)
        if stage.adapter.startswith('replay-'):
            return self.replay(stage,output)
        if stage.adapter=='qualification':
            from .qualification import qualification
            from oeis_learn.experiments.source_snapshot import collect_snapshot
            from .identity import digest
            required={'host.contracts','hip.preflight','policy.compare','image.'+self.args.engine+'.gpu'}
            for engine in self.args.pipeline_engines:
                required.update({'engine.'+engine,'image.'+engine+'.qualification','cpu.'+engine+'.semantic',
                    'operations.'+engine+'.retirement','operations.'+engine+'.workflow'})
            report=qualification({k:v for k,v in self.receipts.items() if k!='qualification'},source=digest(collect_snapshot(self.root)),required=required)
            write(output/'qualification.json',report)
            return report
        if stage.adapter=='cost':
            return self.benchmarks.cost(output,cap=stage.seconds,scope='diverse32' if stage.id=='cost.diverse32' else 'transfer-screen')
        if stage.adapter in {'baseline','inference','joint'}:
            return getattr(self.benchmarks,stage.adapter)(output,cap=stage.seconds)
        parts=stage.id.split('.')
        name=parts[1] if parts[0] in {'engine','image','cpu','operations'} else self.args.engine
        host=self.host(name)
        if stage.adapter=='engine':
            facts=host.engine.facts();write(output/'engine.json',dict(facts=facts,resources=host.policy['pipeline_resources']))
            return facts
        if stage.adapter=='image':
            target=parts[2]
            if target=='gpu':host.select_gpu()
            import time
            started=time.monotonic()
            image=self.images.ensure(host,target,output/'build',seconds=stage.seconds)
            # Whole installed package inspection is mandatory for pinned and built images.
            from oeis_learn.container_validation import inspect_current_image
            remaining=min(stage.seconds-(time.monotonic()-started),self.images.cap-self.images.used.get(name,0))
            if remaining<=0:raise TimeoutError('shared build/inspection reservation exhausted')
            inspection_started=time.monotonic()
            try:
                inspect_current_image(host,image,output/'inspection',seconds=min(600,remaining))
            finally:
                self.images.used[name]=self.images.used.get(name,0)+time.monotonic()-inspection_started
            write(output/'image.json',dict(image=image,target=target,engine=name))
            return dict(image=image)
        target='gpu' if stage.adapter=='hip' else 'qualification'
        if stage.adapter in {'policy','native'}:
            from .benchmarks import legacy
            container_stage=legacy(self.root).container_stage
            from .identity import digest
            from oeis_learn.experiments.source_snapshot import collect_snapshot
            source=output/'input';source.mkdir()
            mode='contracts' if stage.adapter=='policy' else 'native'
            write(source/'expected.json',dict(version='pipeline-role/v1',source=digest(collect_snapshot(self.root)),contract=self.contract['identity'],mode=mode,options={'seconds':stage.seconds,'memory_ceiling':host.policy['memory_bytes']}))
            image=self.images.images[(name,target)]
            container_stage(host,image,source,output/'role',mode=mode,seconds=stage.seconds,gpu=False,
                argv=['python','-m','oeis_learn.pipeline_validation.role',mode])
            result=json.loads((output/'role/summary.json').read_text())
            if result.get('status')=='FAIL':
                raise RuntimeError(result.get('error','internal role failure'))
            if stage.adapter=='native':
                from .benchmarks import native_component
                component=native_component(result)
                self.benchmarks.selected_workers=component['workers']
                write(output/'component.json',component)
                return dict(comparison=result,component=component)
            if result.get('status')=='INCONCLUSIVE':
                result['stage_state']='INCONCLUSIVE'
            return result
        image=self.images.images[(name,target)]
        host.image=image
        if stage.adapter=='hip':
            from oeis_learn.container_validation import _stages
            host.select_gpu()
            with host.gpu_lock():
                return _stages(host,'bigint-gpu',output,external_seconds=stage.seconds)[1]
        from oeis_learn.container_validation import _stages
        return _stages(host,'bigint-retirement' if stage.adapter=='retirement' else 'cpu' if stage.adapter=='operations' else 'bigint-cpu',output,external_seconds=stage.seconds)[1]
