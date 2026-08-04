"""TRAIN-only selective recovery eligibility for Exp425."""

from .contracts import EligibilityAssessment, FoldIdentityCounts, GroupEligibilityEvidence
from .eligibility import apply_group_eligibility, evaluate_group_eligibility
from .predecessor import (
    Exp424CompatibilityContract,
    Exp424ImportReport,
    ValidatedExp424Source,
    adapt_imported_result,
    import_exp424_nonterminal,
    inspect_exp424_source,
    open_validated_exp424_source,
)

__all__ = [
    "EligibilityAssessment",
    "FoldIdentityCounts",
    "GroupEligibilityEvidence",
    "apply_group_eligibility",
    "evaluate_group_eligibility",
    "Exp424CompatibilityContract",
    "Exp424ImportReport",
    "ValidatedExp424Source",
    "adapt_imported_result",
    "import_exp424_nonterminal",
    "inspect_exp424_source",
    "open_validated_exp424_source",
]
