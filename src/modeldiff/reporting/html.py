"""Dependency-free standalone HTML report rendering."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modeldiff.result import ModelDiffReport, RegionResult


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def _region_table(regions: list[RegionResult]) -> str:
    if not regions:
        return '<p class="empty">No regions discovered.</p>'
    rows = []
    for region in regions:
        low, high = region.confidence_interval
        status = '<span class="sig">yes</span>' if region.significant else "no"
        rows.append(
            "<tr>"
            f"<td>{escape(region.description)}</td>"
            f"<td>{region.validation_support} ({region.validation_fraction:.1%})</td>"
            f"<td>{region.baseline_mean_loss:.4f}</td>"
            f"<td>{region.candidate_mean_loss:.4f}</td>"
            f"<td class={'bad' if region.validation_delta > 0 else 'good'}>"
            f"{region.validation_delta:+.4f}</td>"
            f"<td>[{low:+.4f}, {high:+.4f}]</td>"
            f"<td>{region.p_value:.4g}</td><td>{region.q_value:.4g}</td><td>{status}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Region</th><th>Validation support</th>'
        "<th>Baseline loss</th><th>Candidate loss</th><th>Delta</th><th>CI</th>"
        "<th>p</th><th>q</th><th>Significant</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_html(report: ModelDiffReport) -> str:
    """Render a complete report with embedded CSS and no external assets."""
    diagnostics = "".join(
        f"<tr><td>{escape(key.replace('_', ' ').title())}</td><td>{value:.4f}</td></tr>"
        for key, value in report.disagreement.items()
    )
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{_display(report.aggregate_metrics['baseline'].get(name))}"
        f"</td><td>{_display(report.aggregate_metrics['candidate'].get(name))}</td></tr>"
        for name in report.aggregate_metrics["baseline"]
    )
    warnings = "".join(f"<li>{escape(item)}</li>" for item in report.warnings) or "<li>None</li>"
    config_rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{escape(str(value))}</td></tr>"
        for key, value in report.configuration.to_dict().items()
    )
    delta = float(report.overall["absolute_delta"])
    verdict = "improvement" if delta < 0 else "regression" if delta > 0 else "no change"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ModelDiff-ML Report</title><style>
:root{{--ink:#17202a;--muted:#667085;--line:#e4e7ec;--bg:#f8fafc;--brand:#3157d5;--good:#087443;--bad:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:40px auto;padding:0 22px}} h1{{font-size:32px;margin:0}} h2{{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.sub{{color:var(--muted);margin-top:4px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.card{{background:white;border:1px solid var(--line);border-radius:10px;padding:18px}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}} .value{{font-size:25px;font-weight:650;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}} th{{font-size:12px;color:var(--muted);text-transform:uppercase}} .table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}
.good{{color:var(--good);font-weight:650}} .bad{{color:var(--bad);font-weight:650}} .sig{{background:#fee4e2;color:var(--bad);padding:2px 7px;border-radius:10px}} .empty{{color:var(--muted)}} details{{margin-top:24px}}
</style></head><body><main><h1>ModelDiff-ML Report</h1><p class="sub">Independent discovery and validation · schema {escape(report.schema_version)}</p>
<div class="cards"><div class="card"><div class="label">Samples</div><div class="value">{report.sample_count:,}</div></div>
<div class="card"><div class="label">Task</div><div class="value">{escape(report.task.replace("_", " "))}</div></div>
<div class="card"><div class="label">Primary metric</div><div class="value">{escape(report.metric)}</div></div>
<div class="card"><div class="label">Overall result</div><div class="value {"good" if delta < 0 else "bad" if delta > 0 else ""}">{verdict}</div></div></div>
<h2>Overview</h2><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<p>Primary delta (candidate loss − baseline loss): <strong>{delta:+.4f}</strong>. Negative means improvement; positive means regression.</p>
<h2>Prediction behavior</h2><div class="table-wrap"><table><tbody>{diagnostics}</tbody></table></div>
<h2>Validated regressions</h2>{_region_table(report.regression_regions)}
<h2>Validated improvements</h2>{_region_table(report.improvement_regions)}
<h2>Warnings</h2><ul>{warnings}</ul>
<details><summary><strong>Methodology and configuration</strong></summary><p>Regions were selected only on the discovery partition, then evaluated unchanged on an independent validation partition. Confidence intervals use percentile bootstrap resampling; p-values use paired sign flips; q-values use Benjamini–Hochberg correction.</p><div class="table-wrap"><table><tbody>{config_rows}</tbody></table></div></details>
</main></body></html>"""
