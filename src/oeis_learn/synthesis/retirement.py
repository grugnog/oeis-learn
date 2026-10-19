"""Read-only source/distribution audit; never imports a retired solver."""

import ast
import importlib.metadata
from pathlib import Path
import re

RETIRED_MODULES = (
    "constant_solver", "grounding", "dixon_solver", "qfnia_solver",
    "mersenne61_filter", "recurrence", "grounding_reference", "grounding_evidence",
)
RETIRED_FUNCTIONS = {
    "grounded_variants", "solve_dixon_bounded", "solve_exact_bounded",
    "solve_integer_constraints", "parse_ast_placeholders", "resolve_program_constants",
}


def audit(package_root, *, project_root=None, installed=False):
    package_root = Path(package_root)
    violations = []
    for name in RETIRED_MODULES:
        if (package_root / "decoder" / (name + ".py")).exists():
            violations.append("retired module: decoder/" + name + ".py")
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in RETIRED_FUNCTIONS:
                violations.append(f"retired algorithm/API: {path.name}:{node.name}")
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.module == "oeis_learn.decoder":
                    names += [node.module + "." + a.name for a in node.names]
            elif isinstance(node, ast.Call):
                fn = node.func
                dynamic = (isinstance(fn, ast.Name) and fn.id == "__import__") or (
                    isinstance(fn, ast.Attribute) and fn.attr == "import_module"
                )
                if dynamic and node.args and isinstance(node.args[0], ast.Constant):
                    names = [node.args[0].value]
            for name in names:
                if isinstance(name, str) and (name.split(".")[0] == "z3" or any(
                    name == "oeis_learn.decoder." + old or name.startswith("oeis_learn.decoder." + old + ".")
                    for old in RETIRED_MODULES
                )):
                    violations.append(f"retired import: {path.name}:{name}")
    if project_root is not None:
        root = Path(project_root)
        for n in ("008_learning_run", "009_recurrence_sprint", "010_overnight_curriculum"):
            if (root / "scripts" / ("run_" + n + ".py")).exists():
                violations.append("retired launch script: " + n)
        for n in (8, 9, 10):
            if (root / "configs" / f"train_run{n:03d}.yaml").exists():
                violations.append(f"retired config: train_run{n:03d}.yaml")
        for rel in ("pyproject.toml", "uv.lock", "docker/dev/install.sh"):
            if re.search(r"\bz3-solver\b", (root / rel).read_text()):
                violations.append("retired dependency: " + rel)
    if installed:
        requires = importlib.metadata.requires("oeis-learn") or []
        if any(re.match(r"z3[-_]solver\b", r, re.I) for r in requires):
            violations.append("installed dependency: z3-solver")
        try:
            importlib.metadata.distribution("z3-solver")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            violations.append("installed Python z3-solver distribution")
    return dict(status="PASS" if not violations else "FAIL", violations=violations)
