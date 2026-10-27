"""Expose the portable workstation CLI from the installed main entry point."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main(argv):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--workspace',type=Path,default=Path.cwd())
    args,remaining=parser.parse_known_args(argv)
    root=args.workspace.resolve()
    script=root/'scripts/foundation_local.py'
    if not script.is_file() or not (root/'Dockerfile').is_file():
        print('ERROR: pipeline requires the project checkout; use --workspace PATH',file=sys.stderr);return 1
    spec=importlib.util.spec_from_file_location('oeis_portable_host',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main(remaining)
