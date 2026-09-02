"""
A `/metrics` endpoint, so this can be watched rather than only inspected.

Everything the system knows about itself is currently answered by asking: open
the dashboard, poll `/stats/trend`, run `app.diagnostics`. That is fine for a
demo and useless for anything left running. There is no way to alert on the
regression rate climbing, no way to graph deviation rate against a deploy, and
no record of what the optimizer was doing an hour ago beyond what happens to be
in the log.

Prometheus' text exposition format is the lowest-dependency way to fix that: it
is plain text, it needs no client library to *produce*, and every monitoring
stack in common use can scrape it.

## What is exported, and why each one

The four numbers that would actually page someone:

  * **deviation rate** -- zero means the optimizer is an expensive no-op, and
    nothing else on the dashboard says so plainly.
  * **regression rate** against what the gate promised -- the drift signal from
    `app.diagnostics`, which is the difference between "it was calibrated" and
    "it is still right".
  * **saved and regressed milliseconds** -- the two halves of whether it is
    earning its keep, kept separate because a net figure hides the trade.
  * **decision cache hit rate** -- near zero means the planning cost the cache
    exists to amortise is being paid in full.

Plus the model's identity and the count of blocked queries, so a graph can be
attributed to a particular deployment.

Deliberately no per-query series. Cardinality is how monitoring falls over, and
a system that fingerprints every ad-hoc statement it is handed would emit an
unbounded number of label values.
"""

from __future__ import annotations


def _line(name: str, value, help_text: str, kind: str = "gauge", labels: str = "") -> list[str]:
    """One metric in Prometheus' text exposition format."""
    if value is None:
        return []
    if isinstance(value, bool):
        value = int(value)
    return [
        f"# HELP lqo_{name} {help_text}",
        f"# TYPE lqo_{name} {kind}",
        f"lqo_{name}{labels} {value}",
    ]


def render(
    trend: dict | None,
    diagnostics_report: dict | None,
    model_status: dict | None,
) -> str:
    """
    The exposition text for one scrape.

    Takes already-computed reports rather than fetching them, so the endpoint
    decides what a scrape is allowed to cost and this stays testable without a
    database.
    """
    out: list[str] = []
    quality = (trend or {}).get("decision_quality") or {}
    overall = (trend or {}).get("overall") or {}

    deviations = (
        quality.get("deviated_win", 0)
        + quality.get("deviated_wash", 0)
        + quality.get("deviated_loss", 0)
    )
    runs = deviations + quality.get("held_correct", 0) + quality.get("held_missed", 0)

    out += _line("runs_total", runs, "Matched served-vs-native runs in the reporting window", "counter")
    out += _line(
        "deviation_rate",
        (deviations / runs) if runs else None,
        "Share of runs where a hinted plan was served instead of PostgreSQL's",
    )
    out += _line(
        "regression_rate",
        (quality.get("deviated_loss", 0) / deviations) if deviations else None,
        "Share of deviations that turned out slower than native",
    )
    out += _line("saved_ms_total", quality.get("saved_ms"), "Milliseconds saved by deviations", "counter")
    out += _line(
        "regressed_ms_total",
        quality.get("regression_ms"),
        "Milliseconds added by deviations that lost",
        "counter",
    )
    out += _line(
        "missed_ms_total",
        quality.get("missed_ms"),
        "Milliseconds available on runs where the optimizer held native",
        "counter",
    )
    out += _line(
        "native_ms_total", overall.get("native_total_ms"), "Native latency over the window", "counter"
    )
    out += _line(
        "served_ms_total", overall.get("served_total_ms"), "Served latency over the window", "counter"
    )

    report = diagnostics_report or {}
    gate = report.get("gate") or {}
    out += _line(
        "gate_consults_uncertainty",
        gate.get("consults_uncertainty"),
        "1 when the gate spends the model's uncertainty; 0 when confidence_z is 0",
    )
    out += _line(
        "gate_calibration_queries",
        gate.get("calibrated_on_n_queries"),
        "How many queries the deployed gate was chosen on",
    )
    out += _line(
        "findings_total",
        len(report.get("warnings") or ()),
        "Open findings from /model/diagnostics",
    )
    out += _line(
        "findings_high_total",
        sum(1 for w in report.get("warnings") or () if w.get("severity") == "high"),
        "Open high-severity findings from /model/diagnostics",
    )

    status = model_status or {}
    cache = status.get("decision_cache") or {}
    out += _line(
        "decision_cache_hit_rate",
        cache.get("hit_rate"),
        "Share of decisions served from cache rather than re-derived",
    )
    out += _line(
        "blocked_queries",
        len((status.get("regression_guard") or {}).get("blocked_queries") or ()),
        "Queries the regression guard is currently refusing to deviate on",
    )
    out += _line(
        "pending_feedback_rows",
        status.get("rows_since_last_training"),
        "Trainable executions logged since the deployed model was trained",
    )
    # A constant series carrying the deployed version as a label, which is the
    # usual way to annotate a graph with what was running.
    version = status.get("current_version")
    if version:
        out += _line(
            "model_info", 1, "Deployed model version", labels=f'{{version="{version}"}}'
        )

    return "\n".join(out) + "\n"
