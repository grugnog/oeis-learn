"""Actual isolated installed processes for CPU acceptance; never container/HIP evidence."""

from pathlib import Path
import sys


class LocalProcessEngine:
    def __init__(self, root, pid=None):
        self.root = Path(root)
        self.pid = pid

    def command(self, **kw):
        mounts = {dest: source for source, dest, _ in kw["mounts"]}
        channel = next(
            Path(source).parent for source, dest, _ in kw["mounts"] if dest.startswith("/inputs/")
        )
        args = dict(inputs=str(channel), output=mounts["/output/discovery"])
        for name, mount in dict(
            authority="/authority",
            accounting="/accounting",
            control="/control",
            state_root="/learning-state",
            predecessor_state="/predecessor-state",
            component_state="/component-state",
        ).items():
            if mount in mounts:
                args[name] = (
                    str(Path(mounts[mount]) / "supervisor.json")
                    if name == "authority"
                    else mounts[mount]
                )
        operation = kw["argv"][-1]
        code = (
            "from pathlib import Path;import os;os.environ.update("
            + repr(
                {
                    k: v
                    for k, v in kw["environment"].items()
                    if k in {"OEIS_DISCOVERY_PURPOSE", "OEIS_DISCOVERY_TRACK"}
                }
            )
            + ");from oeis_learn.pipeline_validation.progressive_role import execute;"
        )
        if self.pid:
            code += "Path(" + repr(str(self.pid)) + ").write_text(str(os.getpid()));"
        code += "\nargs={k:Path(v) for k,v in " + repr(args) + ".items()}\n"
        code += "try:\n r=execute(" + repr(operation) + ",**args)\n"
        code += (
            "except Exception as exc:\n from contextlib import nullcontext\n from oeis_learn.pipeline_validation.progressive_evidence import failure\n from oeis_learn.generalization.progressive.storage import owned_guard\n from oeis_learn.generalization.progressive.contracts import load\n guard=owned_guard(load(args['authority'],maximum=32<<20),output=args['output'],state_root=args.get('state_root')) if 'authority' in args else None\n with guard.activate() if guard else nullcontext():r=failure(args['output'],"
            + repr(operation)
            + ",exc,producer='installed-local-process')\n"
        )
        code += "raise SystemExit(0 if r['mechanical_status']=='COMPLETE' else 2 if r['mechanical_status'] in {'PARTIAL','BLOCKED'} else 1)"
        return [sys.executable, "-c", code]

    def remove(self, *args, **kwargs):
        pass
