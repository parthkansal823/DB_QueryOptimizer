const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    // A 400 from /query/analyze or /query/optimize carries a structured
    // explanation of what is wrong with the SQL -- the offending token, a
    // caret under it, and often a corrected statement. Flattening that into
    // `400 Bad Request: {"detail":{...}}` and showing it as a string throws
    // away the whole point of producing it.
    let sqlError = null;
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed.detail === "object" && parsed.detail !== null) {
        sqlError = parsed.detail;
      }
    } catch {
      // Not JSON. Fall through to the plain-text error below.
    }
    const err = new Error(
      sqlError?.message ?? `${res.status} ${res.statusText}: ${body}`,
    );
    err.status = res.status;
    err.sqlError = sqlError;
    throw err;
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

export function fetchModelDiagnostics() {
  return request("/model/diagnostics");
}

export function fetchDatabaseAdvisor(includeStatistics = false) {
  return request(`/dbadvisor?include_statistics=${includeStatistics}`);
}

export function fetchExperimentStatus() {
  return request("/experiment/status");
}

export function startExperiment(body) {
  return request("/experiment/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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
