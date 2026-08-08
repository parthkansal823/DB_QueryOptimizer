import { useEffect, useState } from "react";
import "./App.css";
import {
  analyzeQuery,
  fetchModelStatus,
  fetchTrend,
  triggerRetrain,
  triggerRollback,
} from "./api";
import LatencyChart from "./components/LatencyChart";
import ModelHealth from "./components/ModelHealth";
import PlanComparison from "./components/PlanComparison";
import QueryForm from "./components/QueryForm";
import TrendChart from "./components/TrendChart";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trend, setTrend] = useState(null);
  const [trendError, setTrendError] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [modelBusy, setModelBusy] = useState(false);

  async function loadTrend() {
    try {
      setTrend(await fetchTrend());
      setTrendError(null);
    } catch (err) {
      setTrendError(err.message);
    }
  }

  async function loadModelStatus() {
    try {
      setModelStatus(await fetchModelStatus());
    } catch {
      setModelStatus(null);
    }
  }

  useEffect(() => {
    loadTrend();
    loadModelStatus();
  }, []);

  async function runModelAction(fn) {
    setModelBusy(true);
    try {
      const result = await fn();
      await loadModelStatus();
      return result;
    } catch (err) {
      return { action: "failed", reason: err.message };
    } finally {
      setModelBusy(false);
    }
  }

  async function handleAnalyze(sql) {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeQuery(sql);
      setResult(data);
      loadTrend(); // this run just added rows to plan_execution_log
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const overall = trend?.overall;
  const improvementPct =
    overall?.native_avg_latency_ms && overall?.chosen_avg_latency_ms
      ? (1 - overall.chosen_avg_latency_ms / overall.native_avg_latency_ms) * 100
      : null;

  return (
    <>
      <header className="app-header">
        <h1>Learned Query Optimizer</h1>
        <p>Paste a query, compare native Postgres against the {overall?.selector_mode ?? "..."} plan-selection path.</p>
      </header>

      <section className="card">
        <h2>Query</h2>
        <QueryForm onAnalyze={handleAnalyze} loading={loading} error={error} />
      </section>

      {result && (
        <>
          <section className="card">
            <h2>Baseline vs. chosen plan</h2>
            <PlanComparison
              baseline={result.baseline}
              chosenPlan={result.chosen_plan}
              selectorMode={result.selector_mode}
              decision={result.decision}
            />
          </section>

          <section className="card">
            <h2>Candidate latencies</h2>
            {result.candidates?.length ? (
              <LatencyChart
                baseline={result.baseline}
                candidates={result.candidates}
                chosenIndex={result.chosen_index}
              />
            ) : (
              <div className="empty-state">No join-order candidates for this query (single table, or unsupported shape).</div>
            )}
          </section>
        </>
      )}

      <section className="card">
        <h2>Model health &amp; learning loop</h2>
        <ModelHealth
          status={modelStatus}
          busy={modelBusy}
          onRetrain={() => runModelAction(triggerRetrain)}
          onRollback={() => runModelAction(triggerRollback)}
        />
      </section>

      <section className="card">
        <h2>Historical accuracy</h2>
        {trendError && <div className="error-banner">{trendError}</div>}
        {overall && (
          <div className="stat-row" style={{ marginBottom: 16 }}>
            <div className="stat-tile">
              <div className="label">Native avg</div>
              <div className="value">{overall.native_avg_latency_ms?.toFixed(1) ?? "-"} ms</div>
            </div>
            <div className="stat-tile">
              <div className="label">Chosen avg</div>
              <div className="value">{overall.chosen_avg_latency_ms?.toFixed(1) ?? "-"} ms</div>
            </div>
            <div className="stat-tile">
              <div className="label">Improvement</div>
              <div className="value" style={{ color: improvementPct > 0 ? "var(--status-good)" : "var(--text-primary)" }}>
                {improvementPct != null ? `${improvementPct.toFixed(1)}%` : "-"}
              </div>
            </div>
            <div className="stat-tile">
              <div className="label">Logged executions</div>
              <div className="value">{(overall.n_native ?? 0) + (overall.n_chosen ?? 0)}</div>
            </div>
          </div>
        )}
        <TrendChart byDay={trend?.by_day} />
      </section>
    </>
  );
}
