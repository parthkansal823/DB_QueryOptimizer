import { useEffect, useState } from "react";
import "./App.css";
import {
  analyzeQuery,
  fetchAdvisor,
  fetchCostModel,
  fetchModelStatus,
  fetchSchema,
  fetchTrend,
  optimizeQuery,
  triggerRetrain,
  triggerRollback,
} from "./api";
import CostModelChart from "./components/CostModelChart";
import CumulativeChart from "./components/CumulativeChart";
import DecisionQuality from "./components/DecisionQuality";
import LatencyChart from "./components/LatencyChart";
import ModelHealth from "./components/ModelHealth";
import OptimizedQuery from "./components/OptimizedQuery";
import Recommendations from "./components/Recommendations";
import ProductionRun from "./components/ProductionRun";
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
  const [production, setProduction] = useState(null);
  const [productionBusy, setProductionBusy] = useState(false);
  const [lastSql, setLastSql] = useState(null);
  const [schemaRecs, setSchemaRecs] = useState(null);
  const [costModel, setCostModel] = useState(null);

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

  async function loadCostModel() {
    try { setCostModel(await fetchCostModel()); } catch { setCostModel(null); }
  }

  useEffect(() => {
    loadTrend();
    loadModelStatus();
    loadSchema();
    loadAdvisor();
    loadCostModel();
  }, []);

  async function handleProductionRun() {
    if (!lastSql) return;
    setProductionBusy(true);
    try {
      setProduction(await optimizeQuery(lastSql));
      loadTrend();
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
      // This run just added rows to plan_execution_log, so every panel that
      // reads that table is now stale.
      loadTrend();
      loadCostModel();
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

          {/* Plan details and candidate latencies were two cards showing the
              same run from two angles, repeating the baseline latency in
              both. One card, one story: what ran, and what else was tried. */}
          <section className="card">
            <h2>Plans considered</h2>
            <PlanComparison
              baseline={result.baseline}
              chosenPlan={result.chosen_plan}
              selectorMode={result.selector_mode}
              decision={result.decision}
            />
            {result.candidates?.length ? (
              <>
                <h3 className="subhead">Every candidate, measured</h3>
                <LatencyChart
                  baseline={result.baseline}
                  candidates={result.candidates}
                  chosenIndex={result.chosen_index}
                  vetoed={result.decision?.fell_back_to_baseline}
                />
              </>
            ) : (
              <div className="empty-state">
                No join-order candidates for this query (single table, or unsupported shape).
              </div>
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
        <h2>Served vs. native &mdash; measured on matched runs</h2>
        {trendError && <div className="error-banner">{trendError}</div>}
        <ServedVsNative trend={trend} />
      </section>

      <section className="card">
        <h2>Decision quality &mdash; was each call the right one?</h2>
        <DecisionQuality quality={trend?.decision_quality} />
      </section>

      <section className="card">
        <h2>Over the run sequence</h2>
        <CumulativeChart runs={trend?.runs} />
      </section>

      <section className="card">
        <h2>Is PostgreSQL&rsquo;s cost model right?</h2>
        <CostModelChart data={costModel} />
      </section>

      {/* A per-day rollup needs more than one day to mean anything. Until
          then the run sequence above is the honest time axis, and a two-point
          line chart would just be decoration. */}
      {trend?.by_day?.length > 1 && (
        <section className="card">
          <h2>Day by day</h2>
          <TrendChart byDay={trend.by_day} />
        </section>
      )}

      <section className="card">
        <h2>Discovered schema</h2>
        <SchemaPanel schema={schema} />
      </section>
    </>
  );
}
