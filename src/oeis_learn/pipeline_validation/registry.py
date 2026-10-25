"""Explicit, static versioned graph; file presence never grants availability."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    id: str
    phase: int
    prerequisites: tuple = ()
    available: bool = True
    seconds: int = 1
    adapter: str = ''
    capability: str = ''
    reason: str = ''


def resolve(stages, phase):
    index = {s.id: s for s in stages}
    if len(index) != len(stages):
        raise ValueError('duplicate stage IDs')
    ordered, visiting, visited = [], set(), set()
    def visit(key):
        if key not in index:
            raise ValueError('unknown prerequisite: ' + key)
        if key in visiting:
            raise ValueError('cycle in stage graph')
        if key in visited:
            return
        visiting.add(key)
        for parent in index[key].prerequisites:
            visit(parent)
        visiting.remove(key); visited.add(key); ordered.append(index[key])
    # Validate the entire registry, even an unselected cycle.
    for key in index:
        visit(key)
    if phase not in {'all-ready','1','2','3','4','5'}:
        raise ValueError('unknown engineering phase')
    wanted = {s.id for s in stages if (s.available if phase == 'all-ready' else s.phase == int(phase))}
    def include(key):
        for parent in index[key].prerequisites:
            if parent not in wanted:
                wanted.add(parent); include(parent)
    for key in list(wanted):
        include(key)
    return [s for s in ordered if s.id in wanted]


def registry(engines=('docker',), primary='docker'):
    stages = [Stage('host.contracts',1,seconds=120,adapter='host')]
    for engine in engines:
        stages.extend([
            Stage('engine.'+engine,1,('host.contracts',),seconds=30,adapter='engine',capability=engine),
            Stage('image.'+engine+'.qualification',1,('engine.'+engine,),seconds=7200,adapter='image'),
            Stage('cpu.'+engine+'.semantic',1,('image.'+engine+'.qualification',),seconds=600,adapter='cpu'),
            Stage('operations.'+engine+'.retirement',5,('image.'+engine+'.qualification',),seconds=1800,adapter='retirement'),
            Stage('operations.'+engine+'.workflow',5,('image.'+engine+'.qualification',),seconds=3600,adapter='operations'),
        ])
    stages += [
        Stage('image.'+primary+'.gpu',2,('engine.'+primary,),seconds=7200,adapter='image'),
        Stage('hip.preflight',2,('image.'+primary+'.gpu',),seconds=600,adapter='hip'),
    ]
    stages += [Stage('policy.compare',3,('cpu.'+primary+'.semantic',),seconds=600,adapter='policy'),
               Stage('native.compare',3,('policy.compare',),seconds=3000,adapter='native')]
    stages += [
        Stage('baseline.calibration',1,('cpu.'+primary+'.semantic','hip.preflight'),seconds=2200,adapter='baseline'),
        Stage('inference.compare',2,('baseline.calibration',),seconds=9000,adapter='inference'),
        Stage('overlap.compare',4,('baseline.calibration','policy.compare','native.compare'),seconds=18000,adapter='joint'),
        Stage('qualification',5,seconds=30,adapter='qualification'),
        Stage('cost.diverse32',5,('cpu.'+primary+'.semantic','hip.preflight'),seconds=1500,adapter='cost'),
        Stage('cost.transfer',5,('cpu.'+primary+'.semantic','hip.preflight'),seconds=1500,adapter='cost'),
    ]
    return stages


def replay_registry(engines=('docker',),primary='docker'):
    stages=[s for s in registry(engines,primary) if s.adapter in {'host','engine','image','cpu','hip'}]
    stages.append(Stage('replay.reference',1,('cpu.'+primary+'.semantic','image.'+primary+'.gpu'),seconds=600,adapter='replay-reference'))
    previous=None
    for name in ['cold','warmup','calibration-0','calibration-1','calibration-2']:
        key='replay.'+name
        parents=(previous,) if previous else ('replay.reference','hip.preflight')
        stages.append(Stage(key,2,parents,seconds=600,adapter='replay-trial'));previous=key
    # Reporting remains independent so a failed/unstarted trial is retained.
    stages.append(Stage('replay.report',2,seconds=30,adapter='replay-report'))
    return stages


def replay_comparison_registry(engines=('docker',),primary='docker'):
    from .replay_comparison import SLOTS
    stages=replay_registry(engines,primary)
    previous='replay.compare.freeze'
    stages.append(Stage(previous,2,('replay.report',),seconds=30,adapter='replay-comparison-freeze'))
    for slot in SLOTS:
        stages.append(Stage(slot['id'],2,(previous,),seconds=600,adapter='replay-comparison-trial'))
        previous=slot['id']
    stages.append(Stage('replay.compare.report',2,seconds=30,adapter='replay-comparison-report'))
    return stages


def replay_batching_registry(engines=('docker',),primary='docker'):
    from dataclasses import replace
    adapters={'replay-reference':'replay-batching-reference','replay-trial':'replay-batching-trial',
        'replay-report':'replay-batching-report','replay-comparison-freeze':'replay-batching-freeze',
        'replay-comparison-trial':'replay-batching-trial','replay-comparison-report':'replay-batching-report'}
    return [replace(s,adapter=adapters.get(s.adapter,s.adapter)) for s in replay_comparison_registry(engines,primary)]


def hardware_registry(engines=('docker',),primary='docker'):
    from .replay import TRIALS
    stages=[s for s in registry(engines,primary) if s.adapter in {'host','engine','image','cpu','hip'}]
    stages.append(Stage('hardware.reference',1,('cpu.'+primary+'.semantic','image.'+primary+'.gpu'),seconds=600,adapter='hardware-reference'))
    stages.extend([
        Stage('hardware.overhead.ordinary',2,('hardware.reference','hip.preflight'),seconds=360,adapter='hardware-role'),
        Stage('hardware.overhead.telemetry',2,('hardware.overhead.ordinary',),seconds=360,adapter='hardware-role'),
        Stage('hardware.measurement',2,('hardware.overhead.telemetry',),seconds=30,adapter='hardware-measurement')])
    previous='hardware.measurement'
    for key in TRIALS:
        stages.append(Stage(key,2,(previous,),seconds=600,adapter='hardware-role'));previous=key
    stages.append(Stage('hardware.baseline',2,(previous,),seconds=30,adapter='hardware-baseline'))
    previous='hardware.baseline'
    for batch in (32,64,128):
        key='hardware.b'+str(batch)
        stages.append(Stage(key,2,(previous,),seconds=1800,adapter='hardware-branch'));previous=key
    stages.append(Stage('hardware.report',2,seconds=30,adapter='hardware-report'))
    return stages


def native_capacity_registry(engines=('docker',),primary='docker'):
    stages=[s for s in registry(engines,primary) if s.adapter in {'host','engine','cpu'} or s.adapter=='image' and s.id.endswith('.qualification')]
    stages.append(Stage('native.capacity',3,('cpu.'+primary+'.semantic',),seconds=4500,adapter='native-capacity'))
    stages.append(Stage('native.capacity.report',3,seconds=30,adapter='native-capacity-report'))
    return stages


def coschedule_registry(engines=('docker',),primary='docker'):
    stages=[s for s in registry(engines,primary) if s.adapter in {'host','engine','image','cpu','hip'}]
    stages.append(Stage('coschedule.reference',1,('cpu.'+primary+'.semantic','image.'+primary+'.gpu'),seconds=600,adapter='coschedule-reference'))
    stages.append(Stage('coschedule.study',4,('coschedule.reference','hip.preflight'),seconds=12500,adapter='coschedule-study'))
    stages.append(Stage('coschedule.report',4,seconds=30,adapter='coschedule-report'))
    return stages


def model_sizing_registry(engines=('docker',),primary='docker'):
    from .model_sizes import PROFILES,TRIALS
    stages=[s for s in registry(engines,primary) if s.adapter in {'host','engine','image','cpu','hip'}]
    stages.append(Stage('sizing.prepare',1,('cpu.'+primary+'.semantic','image.'+primary+'.gpu'),seconds=30,adapter='sizing-prepare'))
    for key in PROFILES:
        stages.append(Stage('sizing.'+key+'.reference',1,('sizing.prepare',),seconds=600,adapter='sizing-reference'))
    for key in PROFILES:
        previous='sizing.'+key+'.reference'
        for trial in TRIALS:
            name=f'sizing.{key}.{trial}'
            stages.append(Stage(name,4,(previous,'hip.preflight'),seconds=1110,adapter='sizing-trial'));previous=name
    stages.append(Stage('sizing.report',4,seconds=30,adapter='sizing-report'))
    return stages


def soak_registry(engines=('docker',),primary='docker'):
    """One combined collection; original adoption is separately reported."""
    stages=[s for s in registry(engines,primary) if s.adapter in
        {'host','engine','image','cpu','hip','retirement','operations','policy','native'}]
    for engine in engines:
        stages.append(Stage('soak.operations.'+engine,5,('image.'+engine+'.qualification',),seconds=180,adapter='soak-operations'))
    stages.append(Stage('soak.prepare',1,('cpu.'+primary+'.semantic','image.'+primary+'.gpu'),seconds=60,adapter='soak-prepare'))
    stages.append(Stage('soak.reference',2,('soak.prepare',),seconds=600,adapter='soak-reference'))
    stages.append(Stage('soak.parity',2,('soak.reference','hip.preflight'),seconds=600,adapter='soak-parity'))
    for batch in (8,16,32):
        previous='soak.parity'
        for trial in ('warmup','calibration-0','calibration-1','calibration-2'):
            key=f'soak.b{batch}.{trial}'
            stages.append(Stage(key,2,(previous,'hip.preflight'),seconds=330,adapter='soak-batch'));previous=key
    stages.append(Stage('soak.freeze',2,('soak.parity',),seconds=60,adapter='soak-freeze'))
    stages.append(Stage('soak.joint',4,('soak.freeze','policy.compare','native.compare','operations.'+primary+'.workflow','operations.'+primary+'.retirement','soak.operations.'+primary),seconds=3600,adapter='soak-joint'))
    stages.append(Stage('soak.observation',4,('soak.joint',),seconds=1200,adapter='soak-observation'))
    stages.append(Stage('soak.persistent',4,('soak.observation',),seconds=7800,adapter='soak-persistent'))
    stages.append(Stage('soak.scorer',5,('soak.persistent','image.'+primary+'.qualification'),seconds=120,adapter='soak-scorer'))
    stages.append(Stage('soak.report',5,seconds=60,adapter='soak-report'))
    return stages
