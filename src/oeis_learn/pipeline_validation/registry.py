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
            Stage('engine.'+engine,1,seconds=30,adapter='engine',capability=engine),
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
        Stage('cost.diverse32',5,seconds=30,adapter='cost'),
        Stage('cost.transfer',5,seconds=30,adapter='cost'),
    ]
    return stages
