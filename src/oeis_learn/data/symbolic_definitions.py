"""Hashed formula assumptions; parseability is not OEIS correspondence evidence."""
from copy import deepcopy
from pathlib import Path
import re
from oeis_learn.discovery.formula_syntax import parse_formula, expression_hash
from oeis_learn.experiments.artifacts import compute_canonical_digest, load_json


def validate_definition(entry):
    fields={'definition_id','sequence_ref','kind','source','expression','expression_sha256','domain','assumptions','parser_version'}
    if not isinstance(entry,dict) or set(entry)!=fields: raise ValueError('definition fields')
    if not isinstance(entry['definition_id'],str) or not entry['definition_id'] or entry['kind']!='CLOSED_FORM' or entry['parser_version']!='bounded-rational/v1': raise ValueError('definition identity/kind/parser')
    ref=entry['sequence_ref']
    if not isinstance(ref,dict) or set(ref)!={'oeis_id','index_scale','index_shift'} or not isinstance(ref['oeis_id'],str) or not re.fullmatch(r'A\d{6}',ref['oeis_id']) or type(ref['index_scale']) is not int or ref['index_scale']!=1 or type(ref['index_shift']) is not int or ref['index_shift']!=0: raise ValueError('canonical definition index convention')
    parse_formula(entry['expression'])
    if entry['expression_sha256']!=expression_hash(entry['expression']): raise ValueError('expression hash mismatch')
    source=entry['source']
    if not isinstance(source,dict) or set(source)!={'reference','revision','content_sha256'} or any(not isinstance(source[k],str) or not source[k] for k in source) or not re.fullmatch(r'sha256:[0-9a-f]{64}',source['content_sha256']): raise ValueError('source provenance')
    domain=entry['domain']
    if not isinstance(domain,dict) or set(domain)!={'integer_only','lower_bound','upper_bound'} or domain['integer_only'] is not True or type(domain['lower_bound']) is not int or (domain['upper_bound'] is not None and type(domain['upper_bound']) is not int): raise ValueError('integer domain')
    if not isinstance(entry['assumptions'],list) or any(not isinstance(a,str) or not a for a in entry['assumptions']): raise ValueError('explicit assumptions required')


class SymbolicDefinitionRegistry:
    def __init__(self,registry_path='data/benchmarks/symbolic_definitions_v1.json'):
        self.registry_path=registry_path; self.definitions_by_oeis_id={}; self.registry_sha256=None
        if Path(registry_path).exists(): self.load_registry(registry_path)

    def load_registry(self,registry_path):
        path=Path(registry_path)
        if path.stat().st_size>1<<20: raise ValueError('registry size')
        data=load_json(path.read_text(encoding='utf-8'))
        if set(data)!={'schema_version','registry_id','registry_sha256','created_at','definitions'} or data['schema_version']!='1.0': raise ValueError('registry version/fields')
        if data['registry_sha256']!=compute_canonical_digest(data,'registry_sha256'): raise ValueError('registry hash mismatch')
        if not isinstance(data['definitions'],list): raise ValueError('definitions list')
        entries={}; ids=set()
        for entry in data['definitions']:
            validate_definition(entry); oeis=entry['sequence_ref']['oeis_id']
            if oeis in entries or entry['definition_id'] in ids: raise ValueError('duplicate definition')
            entries[oeis]=deepcopy(entry);ids.add(entry['definition_id'])
        self.definitions_by_oeis_id=entries; self.registry_sha256=data['registry_sha256']

    def get_definition(self,oeis_id): return deepcopy(self.definitions_by_oeis_id.get(oeis_id))
    def has_definition(self,oeis_id): return oeis_id in self.definitions_by_oeis_id
