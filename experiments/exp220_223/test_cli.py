from __future__ import annotations

import pytest

from experiments.exp220_223.run_exp220_223 import build_parser


@pytest.mark.parametrize(
    "command",
    [
        "freeze-policy",
        "replay-audit",
        "ingest-values",
        "predict-shadow",
        "import-outcomes",
        "evaluate-shadow",
    ],
)
def test_cli_exposes_preregistered_commands(command: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as result:
        parser.parse_args([command, "--help"])
    assert result.value.code == 0

