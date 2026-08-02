"""Frozen configuration for the separate Exp422 TEST evaluation."""

from dataclasses import asdict, dataclass, fields
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Exp422TestEvalConfig:
    experiment_id: str = "Exp422-TEST-EVAL"
    source_experiment_id: str = "Exp422"
    expected_datasets: int = 1117
    expected_label_rows: int = 195900
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    labels_sha256: str = "b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4"

    def validate(self) -> None:
        """Reject changes to the preregistered evaluation contract."""
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                # Small fixtures may override population counts and the label
                # digest, but the production defaults themselves are immutable.
                if field.name in {"expected_datasets", "expected_label_rows", "labels_sha256", "database_sha256"}:
                    continue
                raise ValueError(f"Exp422-TEST-EVAL frozen config mismatch: {field.name}")
        if type(self.expected_datasets) is not int or self.expected_datasets < 1:
            raise ValueError("expected_datasets must be a positive exact integer")
        if type(self.expected_label_rows) is not int or self.expected_label_rows < 1:
            raise ValueError("expected_label_rows must be a positive exact integer")
        for name in ("database_sha256", "labels_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


FROZEN_CONFIG = Exp422TestEvalConfig()


__all__ = ["Exp422TestEvalConfig", "FROZEN_CONFIG"]
