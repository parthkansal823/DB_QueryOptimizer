import { useState } from "react";

// A handful of the backend's Phase 1 workload queries (see
// backend/app/workload.py), so the dashboard is useful without the user
// having to know the schema first.
// Samples chosen to show the mechanism, not just to run. The "correlation
// trap" ones filter on both halves of a functional dependency (city implies
// country; brand implies category), which PostgreSQL estimates as
// sel(a) x sel(b) -- several times too small. That error is what produces a
// wrong join order, and what the learned optimizer gets to fix.
const SAMPLE_QUERIES = [
  {
    label: "TRAP: brand implies category (95% headroom)",
    sql: `SELECT oi.id, p.name
FROM order_items oi
JOIN products p ON oi.product_id = p.id
WHERE p.category = 'electronics' AND p.brand = 'Voltix'`,
  },
  {
    label: "TRAP: city implies country (63% headroom)",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.country = 'IN' AND u.city = 'Mumbai'`,
  },
  {
    label: "TRAP: 6-way join, correlations at both ends",
    sql: `SELECT o.id, u.name, p.name, s.name
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
JOIN product_suppliers ps ON ps.product_id = p.id
JOIN suppliers s ON s.id = ps.supplier_id
WHERE u.city = 'London' AND u.country = 'UK'
  AND p.category = 'home' AND p.brand = 'Hearthware'`,
  },
  {
    label: "TRAP: status correlates with channel",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'cancelled' AND o.channel = 'partner'`,
  },
  {
    label: "CONTROL: single filter, nothing to mis-estimate",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.country = 'US'`,
  },
  {
    label: "CONTROL: 4-way, uncorrelated filter",
    sql: `SELECT o.id, u.name, p.name
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE oi.quantity >= 3`,
  },
];

export default function QueryForm({ onAnalyze, loading, error }) {
  const [sql, setSql] = useState(SAMPLE_QUERIES[0].sql);

  function handleSampleChange(e) {
    const sample = SAMPLE_QUERIES.find((s) => s.label === e.target.value);
    if (sample) setSql(sample.sql);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (sql.trim()) onAnalyze(sql);
  }

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <textarea
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        spellCheck={false}
      />
      <div className="query-form-actions">
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        <select onChange={handleSampleChange} defaultValue={SAMPLE_QUERIES[0].label}>
          {SAMPLE_QUERIES.map((s) => (
            <option key={s.label} value={s.label}>
              {s.label}
            </option>
          ))}
        </select>
      </div>
      {error && <div className="error-banner">{error}</div>}
    </form>
  );
}
