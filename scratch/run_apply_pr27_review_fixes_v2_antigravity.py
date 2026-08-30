#!/usr/bin/env python3
"""Run the PR #27 fix applicator while preserving escaped source literals."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
APPLICATOR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "scratch/apply_pr27_review_fixes_antigravity.py"


def main_antigravity() -> None:
    specification = importlib.util.spec_from_file_location(
        "apply_pr27_review_fixes_antigravity",
        APPLICATOR_ANTIGRAVITY,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("PR27 applicator cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    original_replace_once = module.replace_once

    def replace_once_preserving_escaped_literals_antigravity(
        path: str,
        old: str,
        new: str,
    ) -> None:
        try:
            original_replace_once(path, old, new)
            return
        except RuntimeError as original_error:
            escaped_old = old.replace('"\n"', r'"\n"')
            escaped_new = new.replace('"\n"', r'"\n"')
            if escaped_old == old:
                raise original_error
            original_replace_once(path, escaped_old, escaped_new)

    module.replace_once = replace_once_preserving_escaped_literals_antigravity
    module.main()


if __name__ == "__main__":
    main_antigravity()
