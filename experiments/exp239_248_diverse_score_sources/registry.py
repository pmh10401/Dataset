from __future__ import annotations

from dataclasses import dataclass

from experiments.exp239_248_diverse_score_sources.complexity_signature import ComplexitySignatureDeviation
from experiments.exp239_248_diverse_score_sources.fourier_residual import SmoothFourierNormalResidual
from experiments.exp239_248_diverse_score_sources.masked_patch import MaskedPatchContextError
from experiments.exp239_248_diverse_score_sources.normal_density import NonlinearNormalDensity
from experiments.exp239_248_diverse_score_sources.ordinal_mmd import OrdinalPatternDistributionShift
from experiments.exp239_248_diverse_score_sources.recurrence_dynamics import RecurrenceDynamicsDeviation
from experiments.exp239_248_diverse_score_sources.state_space import RobustStateSpaceInnovation
from experiments.exp239_248_diverse_score_sources.symbolic_motif import SymbolicMotifRarity
from experiments.exp239_248_diverse_score_sources.transformation_consistency import TransformationConsistency
from experiments.exp239_248_diverse_score_sources.wavelet_energy import MultiscaleWaveletEnergyDeviation


@dataclass(frozen=True)
class SourceRegistration:
    experiment: int
    name: str
    minimum_train: int
    model_based: bool
    dependency: int | None = None


SOURCE_REGISTRY = {
    239: SourceRegistration(239, "multiscale_wavelet_energy_deviation", 5, False),
    240: SourceRegistration(240, "ordinal_pattern_distribution_shift", 5, False),
    241: SourceRegistration(241, "symbolic_motif_rarity", 5, False),
    242: SourceRegistration(242, "complexity_signature_deviation", 5, False),
    243: SourceRegistration(243, "recurrence_dynamics_deviation", 5, False),
    244: SourceRegistration(244, "robust_state_space_innovation", 10, False),
    245: SourceRegistration(245, "smooth_fourier_normal_residual", 10, False),
    246: SourceRegistration(246, "transformation_consistency", 20, True),
    247: SourceRegistration(247, "masked_patch_context_error", 20, True),
    248: SourceRegistration(248, "nonlinear_normal_density", 50, True, dependency=247),
}


def registration(value: int | str) -> SourceRegistration:
    if isinstance(value, int) or str(value).isdigit():
        experiment = int(value)
        if experiment not in SOURCE_REGISTRY:
            raise KeyError(f"unknown experiment: {value}")
        return SOURCE_REGISTRY[experiment]
    for item in SOURCE_REGISTRY.values():
        if item.name == str(value):
            return item
    raise KeyError(f"unknown score source: {value}")


def build_source(value: int | str, *, device: str = "cpu"):
    item = registration(value)
    constructors = {
        239: MultiscaleWaveletEnergyDeviation,
        240: OrdinalPatternDistributionShift,
        241: SymbolicMotifRarity,
        242: ComplexitySignatureDeviation,
        243: RecurrenceDynamicsDeviation,
        244: RobustStateSpaceInnovation,
        245: SmoothFourierNormalResidual,
        246: lambda: TransformationConsistency(device=device),
        247: lambda: MaskedPatchContextError(device=device),
        248: lambda: NonlinearNormalDensity(device=device),
    }
    return constructors[item.experiment]()

