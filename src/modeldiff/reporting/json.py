"""JSON helpers retained as a focused reporting module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modeldiff.result import ModelDiffReport


def render_json(report: ModelDiffReport, *, indent: int = 2) -> str:
    """Return a report's stable JSON representation."""
    return report.to_json(indent=indent)
