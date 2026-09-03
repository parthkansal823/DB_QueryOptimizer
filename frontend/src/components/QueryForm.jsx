import { useState } from "react";

import SqlErrorPanel from "./SqlErrorPanel";

// A handful of the backend's Phase 1 workload queries (see
// backend/app/workload.py), so the dashboard is useful without the user
// having to know the schema first.
// Samples chosen to show the mechanism, not just to run. The "correlation
// trap" ones filter on both halves of a functional dependency (city implies
// country; brand implies category), which PostgreSQL estimates as
// sel(a) x sel(b) -- several times too small. That error is what produces a
// wrong join order, and what the learned optimizer gets to fix.
// Sample queries, grouped by what each one is *for*.
//
// Every headroom figure below was measured on this database before the sample
// was offered: native against the best plan the served action space can
// actually reach, three executions each, median. A label quoting a number
// nobody measured would be the same class of claim this project spends its
// time removing.
//
// Two candidates were dropped rather than shipped. A top-N over `suppliers`
// showed 39% headroom on a 1.7 ms query -- a percentage of nothing, and noise
// at that scale. A five-way join that linked `suppliers` to `users` on
// `country` ran for 1.77 s with 0% headroom: that is not a hard query, it is a
// near-cartesian join with no key path, and it would have taught a reader the
// wrong lesson about where the difficulty lives.
const SAMPLE_QUERIES = [
  // ---- Correlation traps: the mis-estimates this schema was built to create.
  // PostgreSQL estimates `WHERE a=x AND b=y` as sel(a) x sel(b), which is only
  // right when the columns are unrelated. Here they are not.
  {
    group: "Correlation traps",
    label: "brand implies category (95% headroom)",
    sql: `SELECT oi.id, p.name
FROM order_items oi
JOIN products p ON oi.product_id = p.id
WHERE p.category = 'electronics' AND p.brand = 'Voltix'`,
  },
  {
    group: "Correlation traps",
    label: "city implies country (63% headroom)",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.country = 'IN' AND u.city = 'Mumbai'`,
  },
  {
    group: "Correlation traps",
    label: "6-way join, correlations at both ends",
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
    group: "Correlation traps",
    label: "status correlates with channel",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'cancelled' AND o.channel = 'partner'`,
  },
  {
    group: "Correlation traps",
    label: "price band implies category (21% headroom)",
    sql: `SELECT oi.id, p.name, p.price
FROM order_items oi
JOIN products p ON p.id = oi.product_id
WHERE p.category = 'home' AND p.price_band = 'budget'`,
  },
  {
    group: "Correlation traps",
    label: "three correlated columns at once (9% headroom)",
    sql: `SELECT o.id, u.name, u.language
FROM orders o
JOIN users u ON u.id = o.user_id
WHERE u.city = 'London' AND u.country = 'UK' AND u.language = 'en'
  AND o.created_at >= now() - interval '90 days'`,
  },

  // ---- Plan shapes the traps above never produce. Aggregation, sorting,
  // semi- and anti-joins and window functions each give the planner a
  // different structure to get wrong, and give the model a different feature
  // profile to score.
  {
    group: "Complex shapes",
    label: "6-way with aggregation — 1754 ms, 48% headroom",
    sql: `SELECT s.name, count(DISTINCT o.id) AS orders
FROM orders o
JOIN users u ON u.id = o.user_id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
JOIN product_suppliers ps ON ps.product_id = p.id
JOIN suppliers s ON s.id = ps.supplier_id
WHERE p.category = 'electronics' AND p.brand = 'Voltix'
GROUP BY s.name
ORDER BY orders DESC
LIMIT 10`,
  },
  {
    group: "Complex shapes",
    label: "DISTINCT over a 4-way join (28% headroom)",
    sql: `SELECT DISTINCT p.brand, p.category
FROM orders o
JOIN users u ON u.id = o.user_id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE u.city = 'Mumbai' AND u.country = 'IN'`,
  },
  {
    group: "Complex shapes",
    label: "GROUP BY with a correlated filter (21% headroom)",
    sql: `SELECT p.category, count(*) AS orders, sum(oi.quantity) AS units
FROM order_items oi
JOIN products p ON p.id = oi.product_id
JOIN orders o ON o.id = oi.order_id
WHERE p.brand = 'Voltix' AND p.category = 'electronics'
GROUP BY p.category`,
  },
  {
    group: "Complex shapes",
    label: "top-N: sort over a real join (20% headroom)",
    sql: `SELECT o.id, u.name, sum(oi.quantity) AS units
FROM orders o
JOIN users u ON u.id = o.user_id
JOIN order_items oi ON oi.order_id = o.id
WHERE u.country = 'IN' AND u.city = 'Mumbai'
GROUP BY o.id, u.name
ORDER BY units DESC
LIMIT 20`,
  },
  {
    group: "Complex shapes",
    label: "semi-join: EXISTS over three tables (8% headroom)",
    sql: `SELECT u.id, u.name
FROM users u
WHERE u.city = 'Mumbai' AND u.country = 'IN'
  AND EXISTS (
      SELECT 1 FROM orders o
      JOIN order_items oi ON oi.order_id = o.id
      JOIN products p ON p.id = oi.product_id
      WHERE o.user_id = u.id AND p.price_band = 'premium'
  )`,
  },
  {
    group: "Complex shapes",
    label: "window function: rank within category (5% headroom)",
    sql: `SELECT p.category, p.name, sum(oi.quantity) AS units,
       rank() OVER (PARTITION BY p.category ORDER BY sum(oi.quantity) DESC) AS rk
FROM order_items oi
JOIN products p ON p.id = oi.product_id
WHERE p.price_band = 'premium' AND p.category = 'home'
GROUP BY p.category, p.name`,
  },
  {
    group: "Complex shapes",
    label: "anti-join: NOT EXISTS (4% headroom)",
    sql: `SELECT p.id, p.name
FROM products p
WHERE p.category = 'electronics' AND p.brand = 'Voltix'
  AND NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.id)`,
  },

  // ---- Controls. Nothing here is mis-estimated, so there is little for a
  // plan-picker to win -- which is the point: a system that "improves" these
  // is reporting noise.
  {
    group: "Controls",
    label: "single filter, nothing to mis-estimate",
    sql: `SELECT o.id, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.country = 'US'`,
  },
  {
    group: "Controls",
    label: "4-way, uncorrelated filter",
    sql: `SELECT o.id, u.name, p.name
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE oi.quantity >= 3`,
  },
];

// Group order is the order the groups first appear above.
const SAMPLE_GROUPS = SAMPLE_QUERIES.reduce((groups, sample) => {
  const existing = groups.find((g) => g.name === sample.group);
  if (existing) existing.items.push(sample);
  else groups.push({ name: sample.group, items: [sample] });
  return groups;
}, []);

export default function QueryForm({ onAnalyze, loading, error, onUseSuggestion }) {
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
        className="query-editor"
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        spellCheck={false}
        aria-label="SQL query"
      />
      <div className="query-form-actions">
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        <select onChange={handleSampleChange} defaultValue={SAMPLE_QUERIES[0].label}>
          {SAMPLE_GROUPS.map((group) => (
            <optgroup key={group.name} label={group.name}>
              {group.items.map((s) => (
                <option key={s.label} value={s.label}>
                  {s.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
      {error && <SqlErrorPanel error={error} onUseSuggestion={onUseSuggestion} />}
    </form>
  );
}
