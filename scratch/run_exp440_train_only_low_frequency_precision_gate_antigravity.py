#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.cli_antigravity import main_antigravity
if __name__=='__main__':
 result=main_antigravity();print(result)
