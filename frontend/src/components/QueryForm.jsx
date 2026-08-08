import { useState } from "react";

// A handful of the backend's Phase 1 workload queries (see
// backend/app/workload.py), so the dashboard is useful without the user
// having to know the schema first.
const SAMPLE_QUERIES = [
  {
    label: "orders x users (country filter)",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.country = 'IN'`,
  },
  {
    label: "order_items x products (price filter)",
    sql: `SELECT oi.id, p.name
FROM order_items oi
JOIN products p ON oi.product_id = p.id
WHERE p.price > 400`,
  },
  {
    label: "3-way: users x orders x order_items",
    sql: `SELECT u.id, o.id, oi.id
FROM users u
JOIN orders o ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
WHERE u.country = 'US'`,
  },
  {
    label: "4-way: full join, country + category",
    sql: `SELECT o.id, u.name, p.name
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE u.country = 'US' AND p.category = 'electronics'`,
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
