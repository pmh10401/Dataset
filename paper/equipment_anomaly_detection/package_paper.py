"""Standalone metadata for the Equipment Anomaly Detection paper rerun.

Frozen origin provenance uses ``static_ast_sha256``: legacy origin modules may
open files or require optional scientific environments at import time.  Their
executable parity is established only after Tasks 4--7 copy the science into
this namespace.
"""

PAPER_NAMESPACE = "paper.equipment_anomaly_detection"
PROVENANCE_VALIDATION_MODE = "static_ast_sha256"
