"""Adopted fixed-prefix amendment; original selector/probe behavior stays sealed."""
import ast
import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path

from oeis_learn.pipeline_validation.identity import seal,digest

POLICY_ID='sha256:7c0be8a12df99cd7c9a9cc3bbefba0b1b55d381a5e1917da63b3e72c9c6580c1'
MEMORY_CEILING=14<<30


def registry():
    value=json.loads(files('oeis_learn.generalization').joinpath('policies/selector-reservoir-amendment.json').read_text())
    if value!=seal(value) or value['identity']!=POLICY_ID:raise ValueError('registered reservoir amendment changed')
    return value


def reservoir_design():
    from .selector_validation import fresh_design
    parent=fresh_design();policy=registry()
    if parent['identity']!=policy['parent_protocol']:raise ValueError('reservoir parent protocol changed')
    cells=[dict(c,draws=policy['draws'].get(c['family'],c['draws'])) for c in parent['cells']]
    if sum(c['draws'] for c in cells)!=7664 or sum(c['draws'] for c in cells)>parent['identity_capacity']:
        raise ValueError('bounded reservoir required')
    return seal(dict(parent,version='pac-selector-structural/v2',parent_protocol=parent['identity'],
        reservoir_amendment=policy['identity'],cells=cells))


def draw(protocol):
    from .structural_draws import draw as original_draw
    if protocol!=reservoir_design():raise ValueError('exact reservoir amendment required')
    return original_draw(protocol,seed=protocol['seed'])


def _protected(text,names):
    tree=ast.parse(text);signatures={}
    # The reporter may append exactly this optional public metadata member.
    # Everything that calculates metrics, eligibility and summaries stays in
    # the protected AST, including the rest of the archive/publish sequence.
    attachment=ast.parse("if (output/'admission.json').is_file():\n    archive.write(output/'admission.json','admission.json')").body[0]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='report':
            for statement in node.body:
                if isinstance(statement,ast.With) and statement.body and ast.dump(statement.body[-1])==ast.dump(attachment):
                    statement.body.pop()
    for name in names:
        candidates=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(candidates)!=1:raise ValueError('protected module function missing/duplicated')
        node=candidates[0]
        if node.decorator_list:raise ValueError('custodian decorator forbidden')
        signature=copy.deepcopy(node);signature.body=[]
        signatures[name]=ast.dump(signature,include_attributes=False)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in names)]
    return ast.dump(tree,include_attributes=False),signatures


def source_compatibility(root,old,current,*,text_reader=None):
    """Preserve every production/build byte except the named custodian edits.

    The two shared modules must preserve their module-level AST and every
    unlisted function. Authorized function signatures cannot acquire decorators
    or default-value side effects. Parent source text is bound by the original
    qualification's source hashes, not a caller's claim of equivalence.
    """
    policy=registry();root=Path(root);reader=text_reader or (lambda p:(root/p).read_text())
    allowed=set(policy['files'])|set(policy['new_files'])
    def critical(name):
        return not (name.startswith('tests/') and name.endswith('.py') or name=='scripts/pac_fresh_support_audit.py')
    changed={p for p in set(old)|set(current) if critical(p) and old.get(p)!=current.get(p)}
    if changed-allowed:raise ValueError('protected source changed: '+', '.join(sorted(changed-allowed)))
    for name in policy['new_files']:
        if name in old or name not in current:raise ValueError('new reservoir module/source mismatch')
    protected={}
    for name,entry in policy['files'].items():
        before=entry['source'];after=reader(name)
        file_digest=lambda s:'sha256:'+hashlib.sha256(s.encode()).hexdigest()
        if file_digest(before)!=entry['sha256'] or old.get(name)!=entry['sha256'] or current.get(name)!=file_digest(after):
            raise ValueError('protected parent/current module bytes changed')
        a,signature_a=_protected(before,entry['functions']);b,signature_b=_protected(after,entry['functions'])
        if a!=b:raise ValueError('protected inference/import code changed: '+name)
        if signature_a!=signature_b:raise ValueError('custodian signature changed: '+name)
        protected[name]=dict(parent=entry['sha256'],current=current[name],protected_ast=digest(a),functions=entry['functions'])
    return seal(dict(version='pac-selector-probe-compatibility/v1',policy=policy['identity'],
        parent_source=digest(old),current_source=digest(current),protected=protected,
        additions={p:current[p] for p in policy['new_files']},
        report_exception='exact optional admission.json archive member only; metric AST unchanged',
        scope='historical probe reuse only; fresh current-source CPU/native and HIP qualification required'))
