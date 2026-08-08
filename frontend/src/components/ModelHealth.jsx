import { useState } from "react";

function VersionRow({ version }) {
  return (
    <tr>
      <td className="mono">{version.version_id}</td>
      <td>{version.promoted ? <span className="badge badge-chosen">serving</span> : ""}</td>
      <td className="mono">
        {version.metrics?.test_mae_ms != null ? `${version.metrics.test_mae_ms.toFixed(1)} ms` : "-"}
      </td>
    </tr>
  );
}

export default function ModelHealth({ status, onRetrain, onRollback, busy }) {
  const [message, setMessage] = useState(null);

  if (!status) return <div className="empty-state">Model status unavailable.</div>;

  const blocked = Object.entries(status.regression_guard?.blocked_queries ?? {});

  async function handle(action, fn) {
    setMessage(null);
    const result = await fn();
    setMessage(`${action}: ${result.action}${result.reason ? ` — ${result.reason}` : ""}`);
  }

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 16 }}>
        <div className="stat-tile">
          <div className="label">Serving</div>
          <div className="value">{status.selector_mode}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Policy</div>
          <div className="value">{status.policy}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Model version</div>
          <div className="value mono" style={{ fontSize: "0.95rem" }}>
            {status.current_version ?? "unversioned"}
          </div>
        </div>
        <div className="stat-tile">
          <div className="label">Unlearned feedback</div>
          <div className="value">{status.rows_since_last_training} rows</div>
        </div>
      </div>

      <div className="query-form-actions" style={{ marginBottom: 16 }}>
        <button type="button" disabled={busy} onClick={() => handle("Retrain", onRetrain)}>
          {busy ? "Working…" : "Retrain & gate"}
        </button>
        <button type="button" className="button-secondary" disabled={busy} onClick={() => handle("Rollback", onRollback)}>
          Roll back
        </button>
      </div>
      {message && <p className="decision-caution" style={{ marginTop: 0 }}>{message}</p>}

      <h3 className="subhead">Regression guard</h3>
      {blocked.length === 0 ? (
        <p className="empty-state">
          No queries blocked — the learned path hasn&rsquo;t measurably regressed on anything yet.
        </p>
      ) : (
        <>
          <p className="decision-caution" style={{ marginTop: 0 }}>
            {blocked.length} quer{blocked.length === 1 ? "y is" : "ies are"} served the native plan:
            their learned-path history is slower than native by more than{" "}
            {(status.regression_guard.tolerance * 100).toFixed(0)}%.
          </p>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Query</th>
                  <th>Native avg</th>
                  <th>Learned avg</th>
                  <th>Ratio</th>
                  <th>Obs.</th>
                </tr>
              </thead>
              <tbody>
                {blocked.map(([id, info]) => (
                  <tr key={id}>
                    <td className="mono">{id}</td>
                    <td className="mono">{info.native_avg_ms.toFixed(1)} ms</td>
                    <td className="mono">{info.chosen_avg_ms.toFixed(1)} ms</td>
                    <td className="mono" style={{ color: "var(--status-critical)" }}>
                      {info.regression_ratio.toFixed(2)}×
                    </td>
                    <td className="mono">{info.n_observations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {status.versions?.length > 0 && (
        <>
          <h3 className="subhead">Model versions</h3>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Status</th>
                  <th>Test MAE</th>
                </tr>
              </thead>
              <tbody>
                {status.versions.map((v) => (
                  <VersionRow key={v.version_id} version={v} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
