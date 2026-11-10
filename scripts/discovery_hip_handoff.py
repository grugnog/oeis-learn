#!/usr/bin/env python3
"""Consolidated spec 013 workstation handoff; system Python and Podman/Docker only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    from oeis_learn.pipeline_validation.progressive_workstation import main
    raise SystemExit(main(ROOT))
