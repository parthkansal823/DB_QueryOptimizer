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

export function fetchCostModel() {
  return request("/stats/cost-model");
}

export function fetchAdvisor() {
  return request("/advisor");
}

// -- runtime configuration --------------------------------------------------

export function fetchSettings() {
  return request("/settings");
}

export function updateSettings(changes) {
  return request("/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
}

export function resetSettings() {
  return request("/settings/reset", { method: "POST" });
}

export function fetchDatabases() {
  return request("/databases");
}

export function testDatabase(url) {
  return request("/databases/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
}

export function addDatabase(name, url) {
  return request("/databases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, url }),
  });
}

export function removeDatabase(name) {
  return request(`/databases/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export function activateDatabase(name) {
  return request(`/databases/${encodeURIComponent(name)}/activate`, { method: "POST" });
}

// -- the user's own workload ------------------------------------------------

export function fetchSavedQueries() {
  return request("/queries");
}

export function validateQuery(sql) {
  return request("/queries/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
  });
}

export function saveQuery(name, sql, description = "") {
  return request("/queries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, sql, description }),
  });
}

export function deleteQuery(name) {
  return request(`/queries/${encodeURIComponent(name)}`, { method: "DELETE" });
}

// -- training ---------------------------------------------------------------

export function fetchTrainStatus() {
  return request("/train/status");
}

export function startTraining(options) {
  return request("/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
}

export function stopTraining() {
  return request("/train/stop", { method: "POST" });
}
