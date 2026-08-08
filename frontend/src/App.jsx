import { useEffect, useState } from "react";
import "./App.css";
import {
  analyzeQuery,
  fetchAdvisor,
  fetchModelStatus,
  fetchRegret,
  fetchSchema,
  fetchTrend,
  optimizeQuery,
  triggerRetrain,
  triggerRollback,
} from "./api";
import LatencyChart from "./components/LatencyChart";
import ModelHealth from "./components/ModelHealth";
import OptimizedQuery from "./components/OptimizedQuery";
import Recommendations from "./components/Recommendations";
import ProductionRun from "./components/ProductionRun";
import RegretChart from "./components/RegretChart";
import SchemaPanel from "./components/SchemaPanel";
import PlanComparison from "./components/PlanComparison";
import QueryForm from "./components/QueryForm";
import ServedVsNative from "./components/ServedVsNative";
import TrendChart from "./components/TrendChart";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trend, setTrend] = useState(null);
  const [trendError, setTrendError] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [modelBusy, setModelBusy] = useState(false);
  const [schema, setSchema] = useState(null);
  const [regret, setRegret] = useState(null);
  const [production, setProduction] = useState(null);
  const [productionBusy, setProductionBusy] = useState(false);
  const [lastSql, setLastSql] = useState(null);
  const [schemaRecs, setSchemaRecs] = useState(null);

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

  async function loadSchema() {
    try { setSchema(await fetchSchema()); } catch { setSchema(null); }
  }

  async function loadAdvisor() {
    try { setSchemaRecs((await fetchAdvisor()).recommendations); } catch { setSchemaRecs(null); }
  }

  async function loadRegret() {
    try { setRegret(await fetchRegret()); } catch { setRegret(null); }
  }

  useEffect(() => {
    loadTrend();
    loadModelStatus();
    loadSchema();
    loadRegret();
    loadAdvisor();
  }, []);

  async function handleProductionRun() {
    if (!lastSql) return;
    setProductionBusy(true);
    try {
      setProduction(await optimizeQuery(lastSql));
      loadRegret();
    } catch (err) {
      setProduction({ reason: `failed: ${err.message}` });
    } finally {
      setProductionBusy(false);
    }
  }

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
      setLastSql(sql);
      setProduction(null);
      loadRegret();
      loadTrend(); // this run just added rows to plan_execution_log
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const overall = trend?.overall;

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
            <h2>Optimized query &mdash; ready to use</h2>
            <OptimizedQuery best={result.best_measured} decision={result.decision} />
          </section>

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

      {result && (
        <section className="card">
          <h2>Production path</h2>
          <ProductionRun result={production} onRun={handleProductionRun} busy={productionBusy} />
        </section>
      )}

      <section className="card">
        <h2>Database optimizations &mdash; fix the cause, not the symptom</h2>
        <Recommendations queryRecs={result?.recommendations} schemaRecs={schemaRecs} />
      </section>

      <section className="card">
        <h2>Cumulative regret</h2>
        <RegretChart regret={regret} />
      </section>

      <section className="card">
        <h2>Discovered schema</h2>
        <SchemaPanel schema={schema} />
      </section>

      <section className="card">
        <h2>Served vs. native &mdash; measured on matched runs</h2>
        {trendError && <div className="error-banner">{trendError}</div>}
        <ServedVsNative trend={trend} />
      </section>

      <section className="card">
        <h2>Over time</h2>
        <TrendChart byDay={trend?.by_day} />
      </section>
    </>
  );
}
