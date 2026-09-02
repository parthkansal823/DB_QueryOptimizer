import { useState } from "react";

// Schema-level fixes, as a list rather than a stack of cards.
//
// Each finding used to render a full card: a severity badge, a paragraph
// explaining why, and a DDL block. That reads well for one finding and badly
// for five, because five unindexed foreign keys produce five nearly identical
// paragraphs and the panel becomes half the page — pushing everything that
// actually varies below it.
//
// One line each, expandable. The line carries what differs between findings
// (which table, which columns, how serious); the explanation and the statement
// are one click away, and identical for every finding of the same kind anyway.

const KIND_LABEL = {
  extended_statistics: "Correlated columns",
  index: "Missing index",
  foreign_key_index: "Unindexed foreign key",
};

function useCopy() {
  const [copied, setCopied] = useState(false);
  return [
    copied,
    async (text) => {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch {
        setCopied(false);
      }
    },
  ];
}

function Recommendation({ rec }) {
  const [open, setOpen] = useState(false);
  const [copied, copy] = useCopy();

  return (
    <div className="rec-row">
      <button
        type="button"
        className="rec-summary"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="collapsible-caret">{open ? "−" : "+"}</span>
        <span className={`badge badge-${rec.severity === "high" ? "vetoed" : "native"}`}>
          {rec.severity}
        </span>
        <span className="rec-kind">{KIND_LABEL[rec.kind] ?? rec.kind}</span>
        <span className="mono rec-target">
          {rec.table}
          {rec.columns?.length ? ` (${rec.columns.join(", ")})` : ""}
        </span>
      </button>

      {open && (
        <div className="rec-detail">
          <p className="decision-caution" style={{ marginTop: 0 }}>{rec.why}</p>
          <div className="sql-block-header">
            <span className="subhead" style={{ margin: 0 }}>{rec.impact}</span>
            <button type="button" className="copy-button" onClick={() => copy(rec.ddl)}>
              {copied ? "Copied" : "Copy DDL"}
            </button>
          </div>
          <pre className="sql-block">{rec.ddl}</pre>
        </div>
      )}
    </div>
  );
}

export default function Recommendations({ queryRecs, schemaRecs }) {
  const all = [...(queryRecs ?? []), ...(schemaRecs ?? [])];
  const [copiedAll, copyAll] = useCopy();

  if (!all.length) {
    return (
      <div className="empty-state">
        Nothing to fix at the schema level &mdash; PostgreSQL&rsquo;s estimates were close
        enough, and no scan was throwing away much work.
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

      <div className="rec-list-head">
        <span className="subhead" style={{ margin: 0 }}>
          {all.length} {all.length === 1 ? "finding" : "findings"}
        </span>
        <button
          type="button"
          className="copy-button"
          onClick={() => copyAll(all.map((r) => r.ddl).join("\n"))}
        >
          {copiedAll ? "Copied" : "Copy all DDL"}
        </button>
      </div>

      <div className="rec-list">
        {all.map((rec, i) => (
          <Recommendation key={`${rec.kind}-${rec.table}-${i}`} rec={rec} />
        ))}
      </div>
    </div>
  );
}
