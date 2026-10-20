"""Read-only source/distribution audit; never imports a retired solver."""

import ast
import copy
import hashlib
import json
import importlib.metadata
from pathlib import Path
import re

RETIRED_MODULES = (
    "constant_solver",
    "grounding",
    "dixon_solver",
    "qfnia_solver",
    "mersenne61_filter",
    "recurrence",
    "grounding_reference",
    "grounding_evidence",
)
RETIRED_FUNCTIONS = {
    "grounded_variants",
    "solve_dixon_bounded",
    "solve_exact_bounded",
    "solve_integer_constraints",
    "parse_ast_placeholders",
    "resolve_program_constants",
}


def function_fingerprint(node):
    node = copy.deepcopy(node)
    node.name = "_retired_body"
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def audit(package_root, *, project_root=None, installed=False):
    package_root = Path(package_root)
    violations = []
    fingerprints = set(
        json.loads(Path(__file__).with_name("retired-fingerprints.json").read_text())["sha256"]
    )
    for name in RETIRED_MODULES:
        if (package_root / "decoder" / (name + ".py")).exists():
            violations.append("retired module: decoder/" + name + ".py")
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        dynamic_names = {"__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                dynamic_names.update(
                    a.asname or a.name for a in node.names if a.name == "import_module"
                )
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in RETIRED_FUNCTIONS
            ):
                violations.append(f"retired algorithm/API: {path.name}:{node.name}")
            if isinstance(node, ast.FunctionDef) and function_fingerprint(node) in fingerprints:
                violations.append(f"copied retired function body: {path.name}:{node.name}")
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.module == "oeis_learn.decoder":
                    names += [node.module + "." + a.name for a in node.names]
            elif isinstance(node, ast.Call):
                fn = node.func
                dynamic = (isinstance(fn, ast.Name) and fn.id in dynamic_names) or (
                    isinstance(fn, ast.Attribute) and fn.attr in {"import_module", "__import__"}
                )
                if dynamic and node.args and isinstance(node.args[0], ast.Constant):
                    names = [node.args[0].value]
            for name in names:
                if isinstance(name, str) and (
                    name.split(".")[0] == "z3"
                    or any(
                        name == "oeis_learn.decoder." + old
                        or name.startswith("oeis_learn.decoder." + old + ".")
                        for old in RETIRED_MODULES
                    )
                ):
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
            text = (root / rel).read_text()
            found = (
                re.search(r"\bz3-solver\b", text)
                if rel != "docker/dev/install.sh"
                else re.search(r"\binstall\b[^\n]*\bz3-solver\b", text)
            )
            if found:
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


def installed_audit():
    """Run under python -I, so checkout/PYTHONPATH cannot hide wheel contents."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    result = audit(root, installed=True)
    if "site-packages" not in root.parts:
        result["violations"].append("audit must import the clean installed distribution")
    for name in RETIRED_MODULES:
        if importlib.util.find_spec("oeis_learn.decoder." + name) is not None:
            result["violations"].append("importable retired module: " + name)
    if importlib.util.find_spec("z3") is not None:
        result["violations"].append("importable Python z3")
    for name in ("synthesis-worker", "certificate-solver"):
        if not (Path("/opt/oeis-build-info") / name).is_file():
            result["violations"].append("missing pinned native artifact: " + name)
    result["status"] = "PASS" if not result["violations"] else "FAIL"
    result["package_root"] = str(root)
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = installed_audit()
    payload = json.dumps(report, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    raise SystemExit(0 if report["status"] == "PASS" else 1)
