"""TRAIN-only selective recovery eligibility for Exp425."""

from .contracts import EligibilityAssessment, FoldIdentityCounts, GroupEligibilityEvidence
from .eligibility import apply_group_eligibility, evaluate_group_eligibility
from .database import DatabaseSnapshotAudit, Exp425DatabaseSnapshot, open_database_snapshot
from .predecessor import (
    Exp424CompatibilityContract,
    Exp424ImportReport,
    ValidatedExp424Source,
    adapt_imported_result,
    import_exp424_nonterminal,
    inspect_exp424_source,
    open_validated_exp424_source,
)
from .runtime import (
    AtomicProgressWriter,
    DatasetRuntimeTrace,
    ExecutionCommand,
    Exp425RunCounts,
    Exp425RunResult,
    Exp425RuntimeSession,
    RuntimePaths,
    RuntimeProgress,
    build_exp425_runtime,
    execute_runtime_command,
    inspect_runtime_output,
    run_selective_population,
)

__all__ = [
    "EligibilityAssessment",
    "FoldIdentityCounts",
    "GroupEligibilityEvidence",
    "apply_group_eligibility",
    "evaluate_group_eligibility",
    "DatabaseSnapshotAudit",
    "Exp425DatabaseSnapshot",
    "open_database_snapshot",
    "Exp424CompatibilityContract",
    "Exp424ImportReport",
    "ValidatedExp424Source",
    "adapt_imported_result",
    "import_exp424_nonterminal",
    "inspect_exp424_source",
    "open_validated_exp424_source",
    "AtomicProgressWriter",
    "DatasetRuntimeTrace",
    "ExecutionCommand",
    "Exp425RunCounts",
    "Exp425RunResult",
    "Exp425RuntimeSession",
    "RuntimePaths",
    "RuntimeProgress",
    "build_exp425_runtime",
    "execute_runtime_command",
    "inspect_runtime_output",
    "run_selective_population",
]
