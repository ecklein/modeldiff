"""Plain-text report rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modeldiff.result import ModelDiffReport, RegionResult


def _format_number(value: float) -> str:
    return f"{value:+.4f}" if value else "0.0000"


def _region_lines(regions: list[RegionResult], total: int) -> list[str]:
    if not regions:
        return ["None discovered."]
    lines: list[str] = []
    for number, region in enumerate(regions, 1):
        marker = "significant" if region.significant else "not significant"
        lines.extend(
            [
                f"{number}. {region.description}",
                f"   Validation population: {region.validation_support:,}/{total:,} "
                f"({region.validation_fraction:.1%})",
                f"   Baseline loss: {region.baseline_mean_loss:.4f}  "
                f"Candidate loss: {region.candidate_mean_loss:.4f}",
                f"   Delta: {_format_number(region.validation_delta)}  "
                f"95% CI: [{region.confidence_interval[0]:+.4f}, "
                f"{region.confidence_interval[1]:+.4f}]",
                f"   p={region.p_value:.4g}  q={region.q_value:.4g} ({marker})",
                "",
            ]
        )
    return lines


def render_console(report: ModelDiffReport) -> str:
    """Render the report as plain text suitable for terminals and logs."""
    overall_delta = float(report.overall["absolute_delta"])
    verdict = "better" if overall_delta < 0 else "worse" if overall_delta > 0 else "unchanged"
    lines = [
        "ModelDiff-ML Report",
        "===================",
        "",
        f"Samples: {report.sample_count:,} "
        f"({report.discovery_count:,} discovery / {report.validation_count:,} validation)",
        f"Validation strategy: {report.split.get('strategy', 'random')}",
        f"Task: {report.task}",
        f"Primary metric: {report.metric}",
        "",
        "Overall",
        "-------",
        f"Baseline:  {report.overall['baseline']:.4f}",
        f"Candidate: {report.overall['candidate']:.4f}",
        f"Difference (candidate - baseline): {_format_number(overall_delta)}",
        f"Candidate is {verdict} overall.",
        "",
        "Prediction behavior",
        "-------------------",
    ]
    for key, value in report.disagreement.items():
        lines.append(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")
    lines.extend(["", "Validated regressions", "---------------------", ""])
    lines.extend(_region_lines(report.regression_regions, report.validation_count))
    lines.extend(["Validated improvements", "----------------------", ""])
    lines.extend(_region_lines(report.improvement_regions, report.validation_count))
    if report.warnings:
        lines.extend(["Warnings", "--------", *[f"- {warning}" for warning in report.warnings]])
    return "\n".join(lines).rstrip()
