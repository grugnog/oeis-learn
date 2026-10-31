"""Bounded generic mechanical software fixtures, never research measurements."""
from oeis_learn.data.bigint_programs import BigintSampler,canonical_family,render,materialize
from oeis_learn.synthesis.reference_parser import parse
from oeis_learn.pipeline_validation.identity import seal,digest
from .refresh import RefreshPlan
from .exclusions import ExclusionIndex

DIAGNOSTIC='mechanical-neutral-expression-diagnostic/v1'


def diagnostic_candidate(source):
    """A declared algebraic diagnostic, still requiring native/reference admission."""
    from oeis_learn.decoder.bigint_codec import encode_body
    program=parse(source)
    result=program.result
    if program.declarations or program.statements or len(result)!=3 or result[0]!='add' or result[2]!=('int','0'):
        raise ValueError('registered neutral-expression fixture required')
    final=render(('program',('return',result[1])))
    return dict(original_source=final,final_source=final,assignment={},tokens=encode_body(final))


def mechanical_inputs(seed=20261016):
    """Fixed 64-draw refresh support and four finite-isolated selection cells.

    The neutral outer expression is an explicit software diagnostic prior. It is
    never attributed to a model proposal or used by the research search path.
    Native admission remains mandatory in canonical preparation and replay.
    """
    sampler=BigintSampler(seed);descriptor=None
    for family in range(0,64,4):
        candidate=sampler.descriptor(family)
        if 2<=len(candidate['parameter_names'])<=4 and candidate['parameter_source'].count('(mul ')>=2:
            program=parse(candidate['parameter_source'],allow_parameters=True)
            parameter=render(('program',('return',('add',program.result,('int','0')))))
            descriptor=seal(dict(candidate,parameter_source=parameter,family=canonical_family(parameter)))
            break
    if descriptor is None:raise ValueError('bounded mechanical descriptor shortfall')
    plan=RefreshPlan([descriptor],seed=seed,capacity=256)
    training,_=plan.take(0,64);index=ExclusionIndex()
    def exclusion(row,role,holdout):
        row=dict(row,group=digest([row['family'],str(row['variant'])]))
        exact=materialize(row)
        return row,dict(id=row['id'],role=role,holdout=holdout,family=row['family'],source=row['source'],values=exact['values'],group=row['group'],variant=str(row['variant']))
    for row in training:
        _,record=exclusion(row,'training','parameter');index=index.add(record)
    variants,_=plan.take(64,192)
    cohorts=[]
    # Fixed candidate counters, roles and order; no retries beyond these bounds.
    for role in ('selection','assessment'):
        for holdout in ('parameter','structural'):
            candidates=variants if holdout=='parameter' else [sampler.sample(i) for i in range(0,1024,64)]
            chosen=None
            for row in candidates:
                try:row,record=exclusion(row,role,holdout)
                except ValueError:continue
                if index.reasons(record):continue
                chosen=row;index=index.add(record);break
            if chosen is None:raise ValueError('bounded mechanical '+role+'/'+holdout+' shortfall')
            cohorts.append(dict(name=role+'-'+holdout,role=role,holdout=holdout,rows=[chosen]))
    # A new target token cap must fit the actual fixed refresh capacity.
    from oeis_learn.decoder.bigint_codec import encode_body
    unique={row['source']:row for row in training}
    if len(unique)<20 or sum(len(encode_body(s))-1 for s in unique)<256:raise ValueError('insufficient mechanical refresh capacity')
    return dict(version='generalization-mechanical-acceptance/v1',seed=seed,descriptors=[descriptor],cohorts=cohorts,
        diagnostic=DIAGNOSTIC,scientific_status='UNMEASURED')


def search_policy(purpose,policy):
    if policy=='model':return 8
    if purpose=='acceptance' and policy==DIAGNOSTIC:return 1
    raise ValueError('unregistered or research diagnostic proposal policy')


def small_recipe(original):
    """Use the already registered 256-wide constructor for fresh smoke updates."""
    import copy
    recipe=copy.deepcopy(original)
    recipe['version']='bigint-bootstrap/v1';recipe.pop('model_profile',None)
    recipe['model'].update(d_model=256,heads=4,encoder_layers=4,decoder_layers=4,ff=1024)
    return recipe
