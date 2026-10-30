"""Independent full-horizon training-only replay admission."""
from dataclasses import dataclass
import json
from .contracts import training_ancestry
from .targets import ground
from oeis_learn.pipeline_validation.identity import seal, digest
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.reference_parser import parse
from oeis_learn.synthesis.admission import reference_identity
from oeis_learn.decoder.bigint_codec import encode_body

@dataclass(frozen=True)
class AdmittedEntry:
    payload: bytes

    def record(self): return json.loads(self.payload)


def admit(task,*,original_source,final_source,assignment,tokens,origin_checkpoint,production,production_identity,order):
    if set(task)!={'id','role','ancestry','source','indices','values'} or task['role']!='training':
        raise ValueError('training-only mechanical task required')
    training_ancestry(task['ancestry'])
    if len(task['indices'])!=100 or len(task['values'])!=100 or len(set(task['indices']))!=100:
        raise ValueError('full 100-term audit required')
    if not origin_checkpoint or not production_identity or type(order) is not int or order<0:
        raise ValueError('origin/runtime/order required')
    if parse(ground(original_source,assignment))!=parse(final_source): raise ValueError('invalid grounding')
    encoded=encode_body(final_source)
    if list(tokens)!=encoded or len(encoded)>=len(encode_body(task['source'])):
        raise ValueError('exact frozen codec target must be strictly shorter')
    oracle=evaluate(task['source'],task['indices'])
    reference=evaluate(final_source,task['indices'])
    native=tuple(production(final_source,tuple(task['indices'])))
    if oracle.outcome!='ok' or reference.outcome!='ok' or not (oracle.values==tuple(task['values'])==reference.values==native):
        raise ValueError('full-horizon production/reference/oracle disagreement')
    record=seal(dict(version='generalization-admitted-entry/v1',task=task['id'],task_identity=digest(task),
                     original_source=original_source,final_source=final_source,assignment=assignment,tokens=encoded,
                     origin_checkpoint=origin_checkpoint,order=order,
                     verification=dict(terms=100,indices=list(task['indices']),values=list(reference.values),
                                       production=production_identity,reference=reference_identity(),scope='finite training horizon')))
    return AdmittedEntry(json.dumps(record,sort_keys=True).encode())
