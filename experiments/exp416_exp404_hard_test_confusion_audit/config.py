"""Frozen Exp416 provenance and cardinality configuration."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Exp416FrozenConfig:
    experiment_id: str = "Exp416"
    expected_datasets: int = 1117
    expected_prediction_datasets: int = 888
    expected_prediction_coordinates: int = 2218
    expected_label_rows: int = 195900
    prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    manifest_sha256: str = "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8"
    labels_sha256: str = "b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"

    def validate(self) -> None:
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp416 frozen config mismatch: {field.name}")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


FROZEN_CONFIG = Exp416FrozenConfig()
