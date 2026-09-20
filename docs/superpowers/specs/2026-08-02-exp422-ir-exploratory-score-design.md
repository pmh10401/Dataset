# Exp422-IR-E Design: review-only score after a failed core gate

Exp422-IR-E is a separately preregistered diagnostic continuation. It keeps
the failed 60-row CPU/MPS core matrix intact, then runs the existing bounded
normal-TRAIN score path for observation only. It never relabels the core stage
as passed and never uses TEST data or metrics.

The two evidence files are intentionally separate: the canonical core manifest
remains valid failed evidence, while the exploratory bundle records the score
stage and the explicit `core_gate_overridden` policy. No production selection is
authorized by either file.
