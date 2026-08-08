import { useState } from "react";

const KIND_LABEL = {
  extended_statistics: "Correlated columns",
  index: "Missing index",
  foreign_key_index: "Unindexed foreign key",
};

function Recommendation({ rec }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(rec.ddl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="rec-card">
      <div className="rec-head">
        <span className={`badge badge-${rec.severity === "high" ? "vetoed" : "native"}`}>
          {rec.severity}
        </span>
        <strong>{KIND_LABEL[rec.kind] ?? rec.kind}</strong>
        <span className="mono" style={{ color: "var(--text-muted)" }}>
          {rec.table}
          {rec.columns?.length ? ` (${rec.columns.join(", ")})` : ""}
        </span>
      </div>

      <p className="decision-caution" style={{ marginTop: 6 }}>{rec.why}</p>

      <div className="sql-block-header">
        <span className="subhead" style={{ margin: 0 }}>{rec.impact}</span>
        <button type="button" className="copy-button" onClick={copy}>
          {copied ? "Copied" : "Copy DDL"}
        </button>
      </div>
      <pre className="sql-block">{rec.ddl}</pre>
    </div>
  );
}

export default function Recommendations({ queryRecs, schemaRecs }) {
  const all = [...(queryRecs ?? []), ...(schemaRecs ?? [])];

  if (!all.length) {
    return (
      <div className="empty-state">
        Nothing to fix at the schema level — PostgreSQL&rsquo;s estimates were close enough, and
        no scan was throwing away much work.
      </div>
    );
  }

  return (
    <div>
      <p className="decision-caution" style={{ marginTop: 0 }}>
        A hint fixes one query. These fix <em>why</em> the hint was needed, so every query using
        these columns gets faster. Check them before you run them: <code>CREATE INDEX</code> takes
        locks and disk space, so nothing here runs on its own.
      </p>
      {all.map((rec, i) => (
        <Recommendation key={`${rec.kind}-${rec.table}-${i}`} rec={rec} />
      ))}
    </div>
  );
}
