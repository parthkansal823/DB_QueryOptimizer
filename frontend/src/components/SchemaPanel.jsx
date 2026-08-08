export default function SchemaPanel({ schema }) {
  if (!schema) return <div className="empty-state">Schema unavailable.</div>;

  const inferred = schema.foreign_keys_inferred;

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <div className="stat-tile">
          <div className="label">Tables</div>
          <div className="value">{schema.n_tables}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Join edges</div>
          <div className="value">{schema.n_foreign_keys}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Total rows</div>
          <div className="value">{schema.total_rows?.toLocaleString()}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Edges from</div>
          <div className="value" style={{ fontSize: "1.1rem" }}>
            {inferred ? "naming" : "declared FKs"}
          </div>
        </div>
      </div>

      {inferred && (
        <p className="decision-caution" style={{ marginTop: 0 }}>
          This database declares no foreign keys, so join edges were inferred from column
          naming (<code>x_id → x.id</code>). Inferred edges are guesses — a wrong one costs a
          slow query, never a wrong answer.
        </p>
      )}

      <h3 className="subhead">Largest tables</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Table</th>
              <th>Rows</th>
              <th>Share</th>
            </tr>
          </thead>
          <tbody>
            {(schema.largest_tables ?? []).map((t) => {
              const share = schema.total_rows ? (t.rows / schema.total_rows) * 100 : 0;
              return (
                <tr key={t.table}>
                  <td className="mono">{t.table}</td>
                  <td className="mono">{t.rows.toLocaleString()}</td>
                  <td>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${Math.max(share, 0.5)}%` }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
