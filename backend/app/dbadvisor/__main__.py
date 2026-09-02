"""
`python -m app.dbadvisor` -- what is wrong with this database, in one pass.

Usage:
    python -m app.dbadvisor                # everything, including statistics
    python -m app.dbadvisor --no-statistics  # catalogue reads only, no ANALYZE
    python -m app.dbadvisor --json         # machine-readable
"""

from __future__ import annotations

import argparse
import json

from app.db import get_cursor
from app.dbadvisor import count_findings, report


def _format(findings: dict) -> str:
    lines: list[str] = []

    stats = findings.get("extended_statistics") or []
    if stats:
        lines += ["", "=== correlated columns the planner does not know about ===",
                  "    PostgreSQL multiplies the selectivities of separate filters. Where",
                  "    the columns are related that estimate is badly wrong, and every plan",
                  "    built from it is built for the wrong number of rows.", ""]
        for s in stats:
            if s.get("measured"):
                lines.append(
                    "  %s(%s): estimate off by %.0fx -> %.1fx"
                    % (s["table"], ", ".join(s["columns"]), s["qerror_before"], s["qerror_after"])
                )
            else:
                lines.append("  %s(%s): %s" % (s["table"], ", ".join(s["columns"]), s["reason"]))
            lines.append("      %s;" % s["definition"])

    config = findings.get("server_config") or {}
    if config.get("findings"):
        lines += ["", "=== server settings ===",
                  "    machine memory: %s" % config.get("machine_memory"), ""]
        for f in config["findings"]:
            lines.append("  %s: %s -> %s" % (f["setting"], f["current"], f["suggested"]))
            lines.append("      %s" % f["note"])
            lines.append("      %s" % f["fix"])

    idx = findings.get("indexes") or {}
    if any(idx.get(k) for k in ("unused", "duplicates", "redundant")):
        lines += ["", "=== indexes that are not earning their keep ==="]
        for u in idx.get("unused", []):
            lines.append("  unused    %s on %s (%s bytes)%s"
                         % (u["index"], u["table"], f"{u['size_bytes']:,}",
                            "  [also enforces uniqueness]" if u["enforces_uniqueness"] else ""))
        for d in idx.get("duplicates", []):
            lines.append("  duplicate %s on %s(%s)"
                         % (", ".join(d["indexes"]), d["table"], ", ".join(d["columns"])))
        for r in idx.get("redundant", []):
            lines.append("  redundant %s on %s -- covered by %s"
                         % (r["index"], r["table"], r["covered_by"]))

    tbl = findings.get("tables") or {}
    if any(tbl.get(k) for k in ("stale_statistics", "bloated", "heavily_scanned")):
        lines += ["", "=== table health ==="]
        for t in tbl.get("stale_statistics", []):
            lines.append("  stale stats  %s: %.0f%% of rows changed since ANALYZE  ->  %s"
                         % (t["table"], t["modified_fraction"] * 100, t["fix"]))
        for t in tbl.get("bloated", []):
            lines.append("  dead rows    %s: %.0f%% dead  ->  %s"
                         % (t["table"], t["dead_fraction"] * 100, t["fix"]))
        for t in tbl.get("heavily_scanned", []):
            lines.append("  scanned      %s: %s sequential vs %s index scans"
                         % (t["table"], t["sequential_scans"], t["index_scans"]))

    if not lines:
        return "\nNothing to report -- no findings on this database.\n"
    lines += ["", "Nothing here has been changed. Each line carries the statement that",
              "would apply it; running one is your decision."]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="public")
    parser.add_argument(
        "--no-statistics",
        action="store_true",
        help="skip the correlated-column check, which costs an ANALYZE per candidate",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    with get_cursor() as cur:
        findings = report(cur, schema=args.schema, include_statistics=not args.no_statistics)

    if args.json:
        print(json.dumps(findings, indent=2, default=str))
    else:
        print(_format(findings))
        print(f"\n{count_findings(findings)} findings.")
