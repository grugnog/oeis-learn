"""Safe qualification metadata; never export raw reasons, parameters or traces."""
import ast
from functools import lru_cache
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .identity import digest,confined

STAGES=('image.qualification','cpu.semantic','image.gpu','hip.preflight')
RECEIPT=re.compile(r'execution/qualification/pipeline/stages/('
    + '|'.join(re.escape(s) for s in STAGES)+r')/receipt\.json')
TESTS=re.compile(r'execution/qualification/pipeline/stages/'
    r'(cpu\.semantic/bigint-cpu-regressions|hip\.preflight/bigint-gpu-regressions)/results\.xml')


@lru_cache(maxsize=512)
def _public_functions(module):
    if not re.fullmatch(r'tests(?:\.[A-Za-z_][A-Za-z0-9_]*)+',module):return None
    root=Path(__file__).resolve().parents[3]
    try:
        path=confined(root,module.replace('.','/')+'.py')
        if path.stat().st_size>1<<20:return None
        tree=ast.parse(path.read_text())
    except (ValueError,OSError,SyntaxError,UnicodeError):return None
    return {n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def _test(case,node):
    module=case.get('classname','');name=case.get('name','');function=name.split('[',1)[0]
    functions=_public_functions(module)
    public=(module.replace('.','/')+'.py::'+function) if functions is not None and function in functions else None
    message=node.get('message','');kind=message.split(':',1)[0]
    allowed={'AssertionError','ImportError','FileExistsError','ValueError','KeyError','RuntimeError','TimeoutError','TypeError','OSError'}
    kind=kind if kind in allowed else 'TestFailure'
    code='qualification_test_failed'
    if public and module=='tests.integration.test_generalization_rehearsal':
        if message.startswith("ImportError: cannot import name 'claim_output' from 'oeis_learn.pipeline_validation.generalization_pac_rehearsal'"):
            code='missing_r0_output_claim'
        elif function=='test_installed_rehearsal_all_three_real_learners_and_in_block_resume' and 'FileExistsError' in message:
            code='r0_output_already_exists'
    # Parameter IDs and unknown test names can contain private fixture values.
    messages=dict(qualification_test_failed='A qualification test failed; its raw message remains private.',
        missing_r0_output_claim='The R0 claim_output implementation is missing.',
        r0_output_already_exists='R0 rejected an output directory already created by the canonical launcher.')
    return dict(test=public,case=digest([module,name]),exception=kind,code=code,message=messages[code])


def test_summary(payload,*,origin,sha256,max_failures=64):
    text=payload.decode('utf-8')
    if re.search(r'<!\s*(DOCTYPE|ENTITY)\b',text,re.IGNORECASE):raise ValueError('XML declarations forbidden')
    tree=ET.fromstring(text)
    if tree.tag not in {'testsuites','testsuite'}:raise ValueError('JUnit root required')
    counts=dict(tests=0,failures=0,errors=0,skipped=0);failures=[];omitted=0
    for case in tree.iter('testcase'):
        counts['tests']+=1
        if counts['tests']>10000:raise ValueError('JUnit case bound')
        for tag in ('failure','error','skipped'):
            node=case.find(tag)
            if node is None:continue
            counts[{'failure':'failures','error':'errors','skipped':'skipped'}[tag]]+=1
            if tag!='skipped':
                if len(failures)<max_failures:failures.append(_test(case,node))
                else:omitted+=1
    if not counts['tests']:raise ValueError('JUnit cases required')
    # pytest may include subtests in suite totals without emitting a testcase for
    # each one. Retain both scopes instead of silently losing those executions.
    reported=dict(tests=0,failures=0,errors=0,skipped=0)
    suites=[s for s in tree.iter('testsuite') if not s.findall('testsuite')]
    for suite in suites:
        for key in reported:
            value=suite.get(key)
            if value is None or not re.fullmatch(r'[0-9]{1,7}',value):
                reported=None;break
            reported[key]+=int(value)
        if reported is None:break
    return dict(origin=origin,producer_sha256=sha256,counts=counts,reported_counts=reported,failures=failures,
        count_scope='counts enumerate testcase elements; reported_counts retain suite totals, including unexpanded subtests',
        omitted_failures=omitted,projection='verified test names and counts; parameter IDs, messages and tracebacks omitted')


def progress_lines(text):
    """Project canonical stage labels; QUALIFICATION reasons can contain paths."""
    import json
    result=[]
    for line in text.splitlines():
        if line.startswith('QUALIFICATION '):
            try:value=json.loads(line[len('QUALIFICATION '):])
            except ValueError:continue
            if (isinstance(value,dict) and isinstance(value.get('stage'),str) and isinstance(value.get('status'),str)
                and value['stage'] in STAGES and value['status'] in {'PASS','FAIL','BLOCKED','PARTIAL','UNAVAILABLE'}):
                result.append('qualification: '+value['stage']+' '+value['status'])
        elif line.startswith(('learner: ','calibration: ')) and '/' not in line and '\\' not in line:
            result.append(line[:2048])
    return result[-64:]
