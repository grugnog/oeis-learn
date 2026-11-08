"""Original-input selection and bounded, visible-only native-fit replay packets."""
import math

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.synthesis.admission import VisiblePrompt,source_identity


def timeout_targets(outcomes):
    selected={}
    for key,row in sorted(outcomes.items()):
        case=row['equal_budget']
        if case['complete'] or case['reason']!='resource_exhausted:timeout':continue
        upstream=case['upstream'];g=upstream['generation'];fits=upstream['fitting']['outcomes']
        if (case['candidate_sha256'] is not None or g!={'attempted':8,'completed':8,'expected':8}
                or len(fits)!=8 or 'resource_exhausted' not in fits
                or set(fits)-{'unsat_in_scope','unsupported','invalid_source','resource_exhausted'}):
            raise ValueError('complete decoded no-selection timeout portfolio required')
        selected[key]=fits
    if not 1<=len(selected)<=32:raise ValueError('one to 32 original timeout cases required')
    return selected


def extract(records,prompts,targets):
    visible={p['id']:p for p in prompts};attempts=[]
    if len(visible)!=len(prompts):raise ValueError('unique visible prompts required')
    for case,outcomes in sorted(targets.items()):
        row=records[case];prompt=visible[case]
        if set(prompt)!={'id','indices','values'} or len(prompt['indices'])!=20 or len(prompt['values'])!=20:
            raise ValueError('exact visible-only prompt required')
        vp=VisiblePrompt(tuple(prompt['indices']),tuple(prompt['values']))
        details=row['details']['correct']['details']
        if row['selections']['equal_budget'] is not None or len(details)!=8 or len(outcomes)!=8:
            raise ValueError('original no-selection eight-attempt portfolio required')
        for position,d in enumerate(details):
            clock=d['fitting']['clock'];allowance=clock['deadline']-clock['admitted']
            if (type(d['position']) is not int or d['position']!=position or type(d['seed']) is not int
                    or not -(2**63)<=d['seed']<2**64 or d['generation']['complete'] is not True
                    or not isinstance(d['source'],str) or len(d['source'].encode())>65536
                    or source_identity(d['source'])!=d['source_sha256']
                    or d['fitting']['outcome']!=outcomes[position]
                    or not math.isfinite(allowance) or not 0<allowance<=4.000001):
                raise ValueError('changed or incomplete original fitting attempt')
            evidence=(d.get('native') or {}).get('evidence') or {}
            if evidence and (evidence['source_sha256']!=d['source_sha256'] or evidence['prompt_sha256']!=vp.sha256):
                raise ValueError('original source/prompt evidence mismatch')
            attempts.append(dict(case=case,position=position,seed=d['seed']%2**32,
                source=d['source'],source_sha256=d['source_sha256'],prompt_sha256=vp.sha256,
                service_seconds=min(4.,allowance),original_outcome=d['fitting']['outcome'],
                original_clock=clock,original_native={k:v for k,v in (d.get('native') or {}).items()
                    if k in {'outcome','reason','elapsed_seconds','timing','searches','completed_bounds','phase'}},
                original_resource=evidence.get('resource_sha256')))
    return seal(dict(version='pac-fit-replay-packet/v1',prompts=[visible[k] for k in sorted(targets)],attempts=attempts))


def core_amendment(name):
    import json
    from importlib.resources import files
    policies={'native-fit-fastpaths/v1':'fit-core-amendment.json',
              'native-fit-reliability/v2':'fit-reliability-amendment.json'}
    if name not in policies:raise ValueError('unknown native fit engineering policy')
    value=json.loads(files('oeis_learn.generalization').joinpath('policies/'+policies[name]).read_text())
    if (set(value)!={'version','name','files','identity'} or seal(value)!=value
            or value['version']!='pac-fit-core-amendment/v1' or value['name']!=name):
        raise ValueError('registered native core amendment changed')
    return value


def check_core(original,current,*,amendment=None):
    def relevant(k):
        return (k.startswith(('src/oeis_learn/synthesis/','crates/oeis_synthesis_core/',
                'crates/oeis_bigint_runtime/','crates/oeis_wasm_evaluator/'))
            or k in {'Dockerfile','pyproject.toml','uv.lock','src/oeis_learn/fresh/bigint_worker.py',
                     'src/oeis_learn/decoder/bigint_codec.py'})
    old={k:v for k,v in original.items() if relevant(k)}
    new={k:v for k,v in current.items() if relevant(k)}
    if amendment is not None:
        changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
        if (seal(amendment)!=amendment or amendment['version']!='pac-fit-core-amendment/v1'
                or amendment['name'] not in {'native-fit-fastpaths/v1','native-fit-reliability/v2'} or changed!=set(amendment['files'])
                or any(set(v)!={'parent','current'} or old.get(k)!=v['parent'] or new.get(k)!=v['current']
                       for k,v in amendment['files'].items())):
            raise ValueError('native core changes differ from registered engineering amendment')
        return
    if not old or old!=new:
        changed=[k for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)]
        raise ValueError('fitter/core/build source differs from original probe run: '+', '.join(changed))


def validate_packet(packet):
    if (set(packet)!={'version','prompts','attempts','identity'} or seal(packet)!=packet
            or packet['version']!='pac-fit-replay-packet/v1' or not 1<=len(packet['prompts'])<=32):
        raise ValueError('closed bounded replay packet required')
    prompts={}
    for p in packet['prompts']:
        if set(p)!={'id','indices','values'} or len(p['indices'])!=20 or len(p['values'])!=20 or p['id'] in prompts:
            raise ValueError('unique visible-only replay prompt required')
        prompts[p['id']]=VisiblePrompt(tuple(p['indices']),tuple(p['values'])).sha256
    fields={'case','position','seed','source','source_sha256','prompt_sha256','service_seconds',
            'original_outcome','original_clock','original_native','original_resource'}
    positions={k:set() for k in prompts}
    for row in packet['attempts']:
        if (set(row)!=fields or row['case'] not in prompts or type(row['position']) is not int
                or not 0<=row['position']<8 or row['position'] in positions[row['case']]
                or type(row['seed']) is not int or not 0<=row['seed']<2**32
                or type(row['service_seconds']) not in (int,float) or not 0<row['service_seconds']<=4
                or not isinstance(row['source'],str) or len(row['source'].encode())>65536
                or source_identity(row['source'])!=row['source_sha256'] or row['prompt_sha256']!=prompts[row['case']]):
            raise ValueError('changed replay source/seed/allowance or prompt')
        positions[row['case']].add(row['position'])
    if any(v!=set(range(8)) for v in positions.values()):raise ValueError('complete fixed fitting portfolios required')
