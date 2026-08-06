from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .config_antigravity import Exp436FrozenConfig, FROZEN_CONFIG


@dataclass(frozen=True)
class ExecutionProfileAntigravity:
    experiment_id: str
    run_identity: str
    cli_program: str
    run_parent: Path

    @property
    def output(self) -> Path:
        return self.run_parent / "output_antigravity"

    @property
    def checkpoint(self) -> Path:
        return self.run_parent / "checkpoint_antigravity"

    @property
    def readiness(self) -> Path:
        return self.run_parent / "readiness_antigravity.json"

    @property
    def evaluation(self) -> Path:
        return self.run_parent / "evaluation_antigravity"


EXP436_PROFILE = ExecutionProfileAntigravity(
    "Exp436",
    "exp436-missing-safe-strict-consensus-antigravity-20260806",
    "run_exp436_missing_safe_strict_consensus_antigravity.py",
    Path("/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806"),
)
EXP437_PROFILE = ExecutionProfileAntigravity(
    "Exp437",
    "exp437-metadata-roundtrip-antigravity-20260807",
    "run_exp437_metadata_roundtrip_antigravity.py",
    Path("/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807"),
)
_ALLOWED_PROFILES = (EXP436_PROFILE, EXP437_PROFILE)
_DATA_ROOT = Path("/Users/mac/Documents/Dataset")


def validated_profile_antigravity(profile: object) -> ExecutionProfileAntigravity:
    if type(profile) is not ExecutionProfileAntigravity or profile not in _ALLOWED_PROFILES:
        raise ValueError("frozen execution profile is invalid")
    if (
        not profile.run_parent.is_absolute()
        or profile.run_parent.parent != _DATA_ROOT
        or profile.run_parent.is_symlink()
    ):
        raise ValueError("frozen execution profile path is invalid")
    return profile


def frozen_config_for_profile_antigravity(profile: object) -> Exp436FrozenConfig:
    selected = validated_profile_antigravity(profile)
    return replace(
        FROZEN_CONFIG,
        experiment_id=selected.experiment_id,
        run_identity=selected.run_identity,
    )


def frozen_config_payload_for_profile_antigravity(profile: object) -> dict[str, object]:
    config = frozen_config_for_profile_antigravity(profile)
    return json.loads(canonical_json_bytes(asdict(config)))


def validate_frozen_config_for_profile_antigravity(
    config: object,
    profile: object,
) -> Exp436FrozenConfig:
    expected = frozen_config_for_profile_antigravity(profile)
    if type(config) is not Exp436FrozenConfig or config != expected:
        raise ValueError("frozen config does not match execution profile")
    return config
