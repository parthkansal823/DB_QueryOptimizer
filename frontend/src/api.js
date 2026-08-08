const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export function analyzeQuery(sql) {
  return request("/query/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
  });
}

export function fetchTrend() {
  return request("/stats/trend");
}

export function fetchModelStatus() {
  return request("/model/status");
}

export function triggerRetrain() {
  return request("/model/retrain?force=true", { method: "POST" });
}

export function triggerRollback() {
  return request("/model/rollback", { method: "POST" });
}

export function optimizeQuery(sql) {
  return request("/query/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
  });
}

export function fetchSchema() {
  return request("/schema");
}

export function fetchRegret() {
  return request("/stats/regret");
}

export function fetchAdvisor() {
  return request("/advisor");
}
