import { useState } from "react";

// The instructions the page never had: what to do, in what order, and which
// numbers to believe.
//
// Folded by default and placed below the query box. Open on arrival it took the
// entire first screen and pushed the one control that does anything below the
// fold -- so the page's answer to "what do I do here" was to hide the thing you
// do it with. A one-line prompt beside the heading is enough to find it.
//
// No persistence: since it starts folded either way, remembering that a
// particular browser had already opened it would change nothing.

function Step({ n, title, children }) {
  return (
    <li className="howto-step">
      <span className="howto-number">{n}</span>
      <div>
        <strong>{title}</strong>
        <div>{children}</div>
      </div>
    </li>
  );
}

export default function HowToUse() {
  const [open, setOpen] = useState(false);

  return (
    <section className="card howto">
      <div className="howto-head">
        <h2 style={{ margin: 0 }}>
          How to use this
          {!open && (
            <span className="collapsible-hint"> what to read, and which numbers to trust</span>
          )}
        </h2>
        <button type="button" className="link-button" onClick={() => setOpen(!open)}>
          {open ? "Hide" : "Show"}
        </button>
      </div>

      {open && (
        <>
          <p>
            This runs your query, then runs about twenty alternative versions of it, and
            reports whether any of them was faster — and whether it was confident enough to
            have picked one for you in production.
          </p>

          <ol className="howto-steps">
            <Step n="1" title="Paste a SELECT and analyze it">
              Anything read-only against the connected database. There are sample queries in
              the box if you would rather not write one. Everything below fills in from that
              one run.
            </Step>

            <Step n="2" title="Read “Faster version of your query” first">
              If a faster alternative exists, this is it — as SQL you can copy and run. The
              hint is a comment, so it is safe to commit: PostgreSQL without the extension
              just ignores it.
            </Step>

            <Step n="3" title="Then “Plans considered” for why">
              What PostgreSQL chose, what the model chose, and whether the pick actually
              turned out faster. The model often declines to switch. That is not a failure —
              it means the predicted win was not big enough to be worth the risk of being
              wrong.
            </Step>

            <Step n="4" title="Check “Database health” once">
              Findings about the database rather than this query: correlated columns the
              planner misjudges, indexes nobody reads, settings left at their defaults. These
              are fixed once and then help every query. Nothing is applied automatically —
              each finding shows the statement and stops.
            </Step>

            <Step n="5" title="Use “Model health” when something looks wrong">
              It answers “should I believe this model”, which nothing else here does. If the
              optimizer is refusing everything, or taking bets it should not, the reason is
              usually listed there.
            </Step>
          </ol>

          <h3 className="subhead">Which numbers to trust</h3>
          <ul className="howto-notes">
            <li>
              <strong>A single run proves very little.</strong> The same query timed twice on
              this database can differ by around 13%. For a real comparison use{" "}
              <em>Settle a comparison</em>, which runs both options many times and says{" "}
              <em>unresolved</em> when the difference is inside the noise.
            </li>
            <li>
              <strong>“Predicted” and “measured” are different claims.</strong> The decision
              note says what the model expected before anything ran. The outcome line says
              what the clock recorded afterwards. When they disagree, the clock is right.
            </li>
            <li>
              <strong>An empty panel is usually an answer.</strong> No alternative plans means
              the query has one sensible shape. No database findings means nothing was found.
            </li>
          </ul>

          <p className="decision-caution">
            The optimizer only deviates when it expects a win <em>and</em> the worst case is
            bounded. On this workload that is roughly 4 queries in 10 — deliberately, because
            an optimizer that is usually right and occasionally three times slower is worse
            than one that does nothing.
          </p>
        </>
      )}
    </section>
  );
}
