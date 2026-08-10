import { useState } from "react";
import { deleteQuery, saveQuery, validateQuery } from "../api";

/**
 * Write a query, check it, and keep it.
 *
 * Saved queries are what the optimizer is trained on, so this is where a user
 * tells it which queries they actually care about -- the alternative was
 * editing `app/workload.py` and restarting the backend.
 */
export default function QueryManager({ queries, onChanged, onError }) {
  const [name, setName] = useState("");
  const [sql, setSql] = useState("");
  const [description, setDescription] = useState("");
  const [check, setCheck] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleValidate() {
    setBusy(true);
    onError(null);
    try {
      setCheck(await validateQuery(sql));
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleSave() {
    setBusy(true);
    onError(null);
    try {
      await saveQuery(name, sql, description);
      setName("");
      setSql("");
      setDescription("");
      setCheck(null);
      await onChanged();
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(queryName) {
    setBusy(true);
    try {
      await deleteQuery(queryName);
      await onChanged();
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <h2>Your queries</h2>
      <p className="setting-help">
        These are the queries the optimizer learns from. Save the ones your application
        actually runs, then train on them below — the model gets better at the workload you
        care about, not the one this project shipped with.
      </p>

      <div className="query-editor">
        <input
          aria-label="Query name"
          placeholder="name (e.g. orders-by-country)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <textarea
          aria-label="SQL"
          rows={6}
          placeholder="SELECT … FROM … JOIN … WHERE …"
          value={sql}
          onChange={(e) => {
            setSql(e.target.value);
            setCheck(null); // a stale "looks good" under edited SQL is a lie
          }}
        />
        <input
          aria-label="Description"
          placeholder="what this query is for (optional)"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <div className="query-actions">
          <button type="button" onClick={handleValidate} disabled={busy || !sql.trim()}>
            Check
          </button>
          <button type="button" onClick={handleSave} disabled={busy || !sql.trim() || !name.trim()}>
            Save
          </button>
        </div>
      </div>

      {check && (
        <p className={check.ok ? "probe-ok" : "probe-fail"}>
          {check.ok ? (
            <>
              Plans cleanly — {check.n_tables} table{check.n_tables === 1 ? "" : "s"} (
              {check.tables.join(", ")}), estimated cost {check.estimated_cost?.toFixed(0)}.
              {/* One table means no join order to choose between, so this query
                  can never produce a candidate and training on it teaches the
                  model nothing. Better said now than after a 20-minute run. */}
              {!check.joins_available && (
                <>
                  {" "}
                  <strong>
                    It only touches one table, so there is no join order to optimize — the
                    optimizer will have nothing to choose between.
                  </strong>
                </>
              )}
            </>
          ) : (
            <>Cannot save: {check.error}</>
          )}
        </p>
      )}

      <ul className="query-list">
        {(queries ?? []).map((query) => (
          <li key={query.name} className="query-item">
            <div>
              <strong>{query.name}</strong>
              {query.description && <span className="query-desc"> — {query.description}</span>}
              <pre>{query.sql}</pre>
            </div>
            <button type="button" onClick={() => handleDelete(query.name)} disabled={busy}>
              Remove
            </button>
          </li>
        ))}
        {(queries ?? []).length === 0 && (
          <li className="empty-state">
            No saved queries yet. Add one above, or train on the built-in benchmark workload.
          </li>
        )}
      </ul>
    </section>
  );
}
