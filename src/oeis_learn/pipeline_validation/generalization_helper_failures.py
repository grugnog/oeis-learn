"""Installed capture and verified public projection of temporary PAC helper errors."""
import ast
import json
from pathlib import Path
import re

from .identity import confined,seal,write
from .generalization_failures import public_failure,_application
from oeis_learn.generalization.contracts import verify_artifact

SOURCE='specs/012-parameterized-adaptive-curriculum/validation/run.py'
STAGES=frozenset({'admission','qualification','calibration','acceptance','rehearsal','baseline','analysis'}|
                 {operation+'-'+arm for operation in ('train','evaluate') for arm in 'ABCD'})
MESSAGE='Temporary campaign helper exception; see the bound source location.'


def capture_helper_failure(target,stage,exc,*,source_root=None):
    """Write a sealed public location and bounded private original exception."""
    from .generalization_pac_export import helper_identity
    if stage not in STAGES:raise ValueError('known helper stage required')
    source_root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[3]
    source=confined(source_root,SOURCE)
    location=None;trace=exc.__traceback__
    while trace is not None:
        if Path(trace.tb_frame.f_code.co_filename).resolve()==source:
            location=dict(function=trace.tb_frame.f_code.co_name,line=trace.tb_lineno)
        trace=trace.tb_next
    record=seal(dict(version='pac-helper-failure/v1',stage=stage,
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',helper=helper_identity(source_root),
        failure=public_failure(exc),location=location,message=MESSAGE))
    target=Path(target)
    write(target/'helper-failure.json',record)
    detail=str(exc)
    write(target/'private/helper-exception.json',dict(kind=type(exc).__name__,error=detail[:65536],truncated=len(detail)>65536))
    return record


def helper_diagnostic(root,row,*,source_root=None):
    """Return only original bound public bytes; an unsafe record becomes an omission."""
    stage=row['stage']
    if stage not in STAGES:raise ValueError('known helper stage required')
    root=Path(root)
    refs=[r for r in row.get('artifacts',[]) if r['path']=='helper-failure.json']
    if not refs and not (root/'helper-failure.json').exists():return None,None
    try:
        if len(refs)!=1:raise ValueError('original helper diagnostic reference required')
        value=json.loads(verify_artifact(root,refs[0],max_bytes=64<<10))
        if (not isinstance(value,dict) or set(value)!={'version','identity','stage','status','helper','failure','location','message'}
            or seal(value)!=value or value['version']!='pac-helper-failure/v1' or value['stage']!=stage
            or value['status'] not in {'FAIL','INTERRUPTED'} or value['status']!=row['status'] or value['message']!=MESSAGE
            or not isinstance(value['failure'],dict) or not isinstance(value['failure'].get('kind'),str)
            or not re.fullmatch(r'[A-Za-z_]{1,64}',value['failure']['kind'])
            or not isinstance(value['helper'],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',value['helper'])):
            raise ValueError('closed helper diagnostic required')
        _application(dict(failure=value['failure']))
        location=value['location']
        if location is not None:
            source_root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[3]
            source=confined(source_root,SOURCE)
            if source.stat().st_size>256<<10:raise ValueError('bounded helper source required')
            functions={node.name for node in ast.parse(source.read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
            if (not isinstance(location,dict) or set(location)!={'function','line'} or not isinstance(location['function'],str)
                or location['function'] not in functions or type(location['line']) is not int or not 1<=location['line']<=100000):
                raise ValueError('public helper source location required')
        return value,None
    except (OSError,ValueError,KeyError,TypeError,UnicodeError,SyntaxError):
        return None,'unverified temporary helper diagnostic omitted: '+stage
