// What is wrong with the database, as opposed to with one query's plan.
//
// Every other panel here is about plan choice: which candidate ran, what the
// model predicted, how the two compared. That is the hard part of the project
// and it is not the first reason most databases are slow. These findings are
// fixed once and then help every query, including ones this system has never
// seen -- correlated columns the planner does not know about, indexes nobody
// reads, statistics describing a table that has since changed, and server
// settings that are defaults rather than decisions.
//
// Nothing here is applied. Each finding carries the statement that would apply
// it, because running DDL against someone's database is their call and not a
// button this dashboard should own.

function Fix({ statement }) {
  if (!statement) return null;
  return <pre className="fix-statement">{statement}</pre>;
}

function Section({ title, blurb, children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h3 style={{ marginBottom: 4 }}>{title}</h3>
      {blurb && <p className="decision-caution" style={{ marginTop: 0 }}>{blurb}</p>}
      {children}
    </div>
  );
}

export default function DatabaseAdvisor({ findings, onRefresh, busy, deep, onDeepChange }) {
  if (!findings) {
    return (
      <div>
        <button type="button" onClick={onRefresh} disabled={busy}>
          {busy ? "Checking…" : "Check the database"}
        </button>
      </div>
    );
  }

  const stats = findings.extended_statistics ?? [];
  const config = findings.server_config ?? {};
  const idx = findings.indexes ?? {};
  const tbl = findings.tables ?? {};
  const nothing = (findings.n_findings ?? 0) === 0;

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <button type="button" onClick={onRefresh} disabled={busy}>
          {busy ? "Checking…" : "Re-check"}
        </button>{" "}
        <label style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={deep} onChange={(e) => onDeepChange(e.target.checked)} />{" "}
          also test for correlated columns
        </label>
        {deep && (
          <p className="decision-caution">
            Each candidate is measured by creating the statistics object, running ANALYZE and
            re-planning, so this is slow on large tables. Nothing is left behind.
          </p>
        )}
      </div>

      {nothing && <p className="decision-note">Nothing to report on this database.</p>}

      {stats.length > 0 && (
        <Section
          title="Correlated columns the planner does not know about"
          blurb={
            "PostgreSQL multiplies the selectivities of separate filters. Where the columns " +
            "are related that estimate is badly wrong, and every plan built from it is built " +
            "for the wrong number of rows — which is the problem this whole project exists to " +
            "work around. CREATE STATISTICS fixes the estimate instead."
          }
        >
          {stats.map((s) => (
            <div key={`${s.table}-${s.columns.join()}`} className="finding finding-high">
              <strong>
                {s.table}({s.columns.join(", ")})
              </strong>
              {s.measured ? (
                <div>
                  estimate off by {s.qerror_before?.toFixed(0)}× → {s.qerror_after?.toFixed(1)}×
                </div>
              ) : (
                <div>{s.reason}</div>
              )}
              <Fix statement={`${s.definition};`} />
            </div>
          ))}
        </Section>
      )}

      {(config.findings ?? []).length > 0 && (
        <Section
          title="Server settings"
          blurb={`Machine memory: ${config.machine_memory}. PostgreSQL's defaults are chosen to start anywhere, not to suit anything.`}
        >
          {config.findings.map((f) => (
            <div key={f.setting} className="finding finding-medium">
              <strong>{f.setting}</strong>: {f.current} → {f.suggested}
              <div>{f.note}</div>
              <Fix statement={f.fix} />
            </div>
          ))}
        </Section>
      )}

      {(idx.unused?.length || idx.duplicates?.length || idx.redundant?.length) > 0 && (
        <Section
          title="Indexes that are not earning their keep"
          blurb="Every index is paid for on each write to its table, and no slow-query report points at one nobody reads."
        >
          {(idx.unused ?? []).map((u) => (
            <div key={u.index} className="finding finding-medium">
              <strong>{u.index}</strong> on {u.table} — never chosen
              {u.enforces_uniqueness && " (also enforces uniqueness — dropping it changes what the database accepts)"}
              <div>{u.note}</div>
            </div>
          ))}
          {(idx.duplicates ?? []).map((d) => (
            <div key={d.indexes.join()} className="finding finding-medium">
              <strong>{d.indexes.join(", ")}</strong> on {d.table}({d.columns.join(", ")}) — identical
            </div>
          ))}
          {(idx.redundant ?? []).map((r) => (
            <div key={r.index} className="finding finding-low">
              <strong>{r.index}</strong> on {r.table} — covered by {r.covered_by}
            </div>
          ))}
        </Section>
      )}

      {(tbl.stale_statistics?.length || tbl.bloated?.length || tbl.heavily_scanned?.length) > 0 && (
        <Section
          title="Table health"
          blurb="A learned optimizer trained against stale statistics is learning to compensate for something an ANALYZE would remove."
        >
          {(tbl.stale_statistics ?? []).map((t) => (
            <div key={`stale-${t.table}`} className="finding finding-high">
              <strong>{t.table}</strong>: {(t.modified_fraction * 100).toFixed(0)}% of rows changed
              since the last ANALYZE
              <Fix statement={`${t.fix};`} />
            </div>
          ))}
          {(tbl.bloated ?? []).map((t) => (
            <div key={`bloat-${t.table}`} className="finding finding-medium">
              <strong>{t.table}</strong>: {(t.dead_fraction * 100).toFixed(0)}% dead rows
              <div>{t.note}</div>
              <Fix statement={`${t.fix};`} />
            </div>
          ))}
          {(tbl.heavily_scanned ?? []).map((t) => (
            <div key={`scan-${t.table}`} className="finding finding-low">
              <strong>{t.table}</strong>: {t.sequential_scans} sequential vs {t.index_scans} index
              scans
              <div>{t.note}</div>
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}
