"""Reproducible Phase 1 caller inventory, never a retirement implementation."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"constant_solver", "grounding", "dixon_solver", "qfnia_solver",
           "mersenne61_filter", "recurrence", "grounding_reference", "grounding_evidence"}


def inventory():
    names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    rows = []
    for name in names:
        path = ROOT / name
        if not path.is_file() or path.suffix not in (".py", ".yaml", ".toml", ".lock", ".sh"):
            continue
        if not name.startswith(("src/", "scripts/", "tests/", "configs/", "docker/", "pyproject", "uv.lock")):
            continue
        text = path.read_text()
        hits = set()
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith("oeis_learn.decoder"):
                        hits.update({part for part in module.split('.') if part in MODULES})
                        if module == "oeis_learn.decoder":
                            hits.update(a.name for a in node.names if a.name in MODULES)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("oeis_learn.decoder"):
                            hits.update(part for part in alias.name.split('.') if part in MODULES)
            # Also catch literal dynamic import paths and test/path inventories.
            hits.update(m for m in MODULES if re.search(r'decoder[./]' + m + r'\b', text))
        if path.suffix == '.py' and name.startswith(('src/', 'tests/')):
            hits.update('data.models.' + model for model in ('ASTSkeleton', 'ConstantSolverResult')
                        if re.search(r'\b' + model + r'\b', text))
        if 'grounded_variants' in text and name.startswith(('src/', 'tests/')):
            hits.add('fresh.search.grounded_variants')
        if path.suffix == '.py' and re.search(r'^\s*(?:import|from) (?:z3|sympy)\b', text, re.M):
            hits.update(lib for lib in ('z3', 'sympy') if re.search(r'^\s*(?:import|from) '+lib+r'\b',text,re.M))
        if name in ('pyproject.toml','uv.lock'):
            hits.update(dep for dep in ('z3-solver','sympy') if dep in text)
        if path.suffix == '.yaml' and (re.search(r'^solver:', text, re.M) or 'enable_z3' in text or path.name in ('train_run008.yaml','train_run009.yaml','train_run010.yaml')):
            hits.add('legacy-solver-config')
        if not hits:
            continue
        disposition = 'migrate'
        replacement = 'synthesis native transport/admission; independent reference for final execution'
        rationale = 'Preserve supported callers and rejection regressions while routing one native fitter'
        if name.startswith('scripts/run_00') or name.startswith('scripts/run_010') or path.name in ('train_run008.yaml','train_run009.yaml','train_run010.yaml'):
            disposition, replacement = 'retire', 'scripts/foundation_run.py versioned bigint recipes after integration'
        if name.startswith('src/oeis_learn/decoder/'):
            disposition, replacement = 'migrate_then_retire', 'qualified Phase 2 native API and independent reference/evidence'
        if hits == {'sympy'}:
            disposition, replacement = 'retain', 'independent certificate/discovery consumer'
            rationale = 'Not an obsolete fitting implementation; do not delete independent acceptance'
        if name == 'src/oeis_learn/discovery/loop_certificate.py':
            rationale = 'Required certificate checker: migrate Z3 dependency before removing Python z3-solver'
        rows.append(dict(path=name, interfaces=sorted(hits), disposition=disposition,
                         replacement=replacement, rationale=rationale,
                         regression='tests/fixtures/bigint/manifest.json; existing owning tests must be ported before deletion',
                         status='retained' if disposition=='retain' else 'pending Phase 4'))
    return dict(version='retirement-inventory/v1', baseline='b96b92ff680a1509d3cede89e9abf0ddb65d2665',
                limitations='Direct imports, dynamic literal module paths and named settings; Phase 4 also audits installed reachability and transitive callers',
                consumers=rows)


if __name__ == '__main__':
    path = ROOT / 'specs/008-bigint-constant-synthesis/retirement-inventory.json'
    path.write_text(json.dumps(inventory(), indent=2) + '\n')
