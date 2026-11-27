#!/usr/bin/env python3
"""Inspect disk use or retire only completed diagnostic timing model tensors."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from oeis_learn.pipeline_validation.discovery_cleanup import main
    raise SystemExit(main(ROOT))
