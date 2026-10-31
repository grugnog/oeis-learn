#!/usr/bin/env python3
"""Assemble explicit generalization inputs without an engine or host packages."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from oeis_learn.pipeline_validation.generalization_inputs import main
if __name__=='__main__':raise SystemExit(main(workspace=ROOT))
