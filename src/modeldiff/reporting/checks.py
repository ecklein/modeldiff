"""Text and standalone HTML rendering for saved-check evaluations."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modeldiff.checks import CheckEvaluationResult


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.6g}" if signed else f"{value:.6g}"


def render_check_console(result: CheckEvaluationResult) -> str:
    """Render a saved-check result for terminals and logs."""
    lines = [
        "ModelDiff-ML Saved Checks",
        "=========================",
        "",
        f"Overall status: {result.overall_status}",
        f"Samples: {result.sample_count:,}",
        f"Task / metric: {result.task} / {result.metric}",
        f"Loss unit: {result.loss_unit}",
        f"Checks: {len(result.outcomes)} total, {result.passed_count} passed, "
        f"{result.failed_count} failed, "
        f"{result.insufficient_data_count} insufficient data",
        "",
    ]
    if not result.outcomes:
        lines.append("No checks evaluated: the saved suite is empty.")
        return "\n".join(lines)
    for outcome in result.outcomes:
        lines.extend(
            [
                f"[{outcome.status.upper()}] {outcome.name} ({outcome.check_id})",
                f"  Region: {outcome.description}",
                f"  Support: {outcome.support_count:,}/{result.sample_count:,} "
                f"({outcome.support_fraction:.1%}); minimum {outcome.min_samples:,}",
                f"  Baseline loss: {_number(outcome.baseline_mean_loss)}  "
                f"Candidate loss: {_number(outcome.candidate_mean_loss)}",
                f"  Delta: {_number(outcome.loss_delta, signed=True)}  "
                f"Tolerance: {outcome.max_loss_increase:.6g}",
                f"  {outcome.explanation}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_check_html(result: CheckEvaluationResult) -> str:
    """Render an escaped, dependency-free saved-check HTML report."""
    rows: list[str] = []
    for outcome in result.outcomes:
        rows.append(
            "<tr>"
            f'<td><span class="status {escape(outcome.status)}">'
            f"{escape(outcome.status)}</span></td>"
            f"<td><strong>{escape(outcome.name)}</strong><br>"
            f'<span class="muted">{escape(outcome.check_id)}</span></td>'
            f"<td>{escape(outcome.description)}</td>"
            f"<td>{outcome.support_count:,} ({outcome.support_fraction:.1%})<br>"
            f'<span class="muted">minimum {outcome.min_samples:,}</span></td>'
            f"<td>{_number(outcome.baseline_mean_loss)}</td>"
            f"<td>{_number(outcome.candidate_mean_loss)}</td>"
            f"<td>{_number(outcome.loss_delta, signed=True)}</td>"
            f"<td>{outcome.max_loss_increase:.6g}</td>"
            f"<td>{escape(outcome.explanation)}</td>"
            "</tr>"
        )
    table = (
        '<p class="empty">No checks evaluated: the saved suite is empty.</p>'
        if not rows
        else '<div class="table-wrap"><table><thead><tr><th>Status</th><th>Check</th>'
        "<th>Fixed region</th><th>Support</th><th>Baseline loss</th>"
        "<th>Candidate loss</th><th>Delta</th><th>Tolerance</th><th>Explanation</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ModelDiff-ML Saved Checks</title><style>
:root{{--ink:#17202a;--muted:#667085;--line:#e4e7ec;--bg:#f8fafc;--good:#087443;--bad:#b42318;--warn:#a15c00}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1300px;margin:40px auto;padding:0 22px}} h1{{font-size:32px;margin:0}} .muted,.sub,.empty{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase}} .value{{font-size:24px;font-weight:650}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}} table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--line)}} th{{font-size:12px;color:var(--muted);text-transform:uppercase}}
.status{{font-weight:700}} .status.pass{{color:var(--good)}} .status.fail{{color:var(--bad)}} .status.insufficient_data{{color:var(--warn)}}
</style></head><body><main><h1>ModelDiff-ML Saved Checks</h1>
<p class="sub">Fixed-condition observed tolerance checks · result schema {escape(result.schema_version)}</p>
<div class="cards"><div class="card"><div class="label">Overall status</div><div class="value">{escape(result.overall_status)}</div></div>
<div class="card"><div class="label">Samples</div><div class="value">{result.sample_count:,}</div></div>
<div class="card"><div class="label">Passed</div><div class="value">{result.passed_count}</div></div>
<div class="card"><div class="label">Failed</div><div class="value">{result.failed_count}</div></div>
<div class="card"><div class="label">Insufficient data</div><div class="value">{result.insufficient_data_count}</div></div></div>
<p>Task: <strong>{escape(result.task)}</strong> · Metric: <strong>{escape(result.metric)}</strong> · Loss unit: <strong>{escape(result.loss_unit)}</strong></p>
{table}
<p class="sub">Statuses compare observed mean loss deltas with configured tolerances. They are not statistical hypothesis-test conclusions.</p>
</main></body></html>"""
