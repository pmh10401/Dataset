"""Checks that the Exp477 HTML dashboard visibly marks completed work."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class _CheckboxParserAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.checkboxes: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "checkbox":
            self.checkboxes.append(values)


def test_dashboard_is_html_and_all_completed_tasks_are_checked() -> None:
    path = Path(__file__).resolve().parents[3] / "docs/EXP477_DASHBOARD_antigravity.html"
    parser = _CheckboxParserAntigravity()
    parser.feed(path.read_text(encoding="utf-8"))

    assert len(parser.checkboxes) >= 6
    assert all("checked" in checkbox for checkbox in parser.checkboxes)
