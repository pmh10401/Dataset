#!/usr/bin/env python3
"""Outside-CWD launcher for the minimal Exp451 command adapter."""
import os, sys
from pathlib import Path
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS","AEON_NUM_THREADS"): os.environ[name]="1"
repo=Path(__file__).resolve().parents[1]
if str(repo) not in sys.path: sys.path.insert(0,str(repo))
from experiments.exp451_rdst_local_gap_qualification_antigravity.cli_antigravity import main_antigravity
if __name__=="__main__": main_antigravity()
