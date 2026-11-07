#!/usr/bin/env python3
"""Read-only PAC selector diagnosis; standard library, no engine required."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from oeis_learn.generalization.selector_audit import main
if __name__ == '__main__':
    raise SystemExit(main())
