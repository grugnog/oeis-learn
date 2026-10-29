#!/usr/bin/env python3
"""Private whole-arm supervisor child; public admission stays in validate --phase 6."""
import argparse
import json
from pathlib import Path
import sys

from foundation_local import Host
from foundation_bigint import execute


def main():
    if len(sys.argv)!=2:
        raise ValueError('one internal arm descriptor required')
    options=argparse.Namespace(**json.loads(Path(sys.argv[1]).read_text()))
    execute(Host(options),options,archive=False)


if __name__=='__main__':
    main()
