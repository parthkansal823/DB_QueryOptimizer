import { useEffect, useState } from "react";
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

export default function Dashboard() {
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
        <p>Paste a query and see how PostgreSQL&rsquo;s plan compares to the one the {overall?.selector_mode ?? "…"} optimizer picks.</p>
      </header>

      <section className="card">
        <h2>Query</h2>
        <QueryForm onAnalyze={handleAnalyze} loading={loading} error={error} />
      </section>

      {result && (
        <>
          <section className="card">
            <h2>Faster version of your query</h2>
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
                <h3 className="subhead">Every plan we tried</h3>
                <LatencyChart
                  baseline={result.baseline}
                  candidates={result.candidates}
                  chosenIndex={result.chosen_index}
                  vetoed={result.decision?.fell_back_to_baseline}
                />
              </>
            ) : (
              <div className="empty-state">
                No alternative join orders for this query — it uses a single table, or a shape
                we do not handle yet.
              </div>
            )}
          </section>
        </>
      )}

      <section className="card">
        <h2>Model health</h2>
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
        <h2>Database fixes &mdash; treat the cause, not the symptom</h2>
        <Recommendations queryRecs={result?.recommendations} schemaRecs={schemaRecs} />
      </section>

      <section className="card">
        <h2>How much time this is saving</h2>
        {trendError && <div className="error-banner">{trendError}</div>}
        <ServedVsNative trend={trend} />
      </section>

      <section className="card">
        <h2>Were the decisions right?</h2>
        <DecisionQuality quality={trend?.decision_quality} />
      </section>

      <section className="card">
        <h2>Time saved, run by run</h2>
        <CumulativeChart runs={trend?.runs} />
      </section>

      <section className="card">
        <h2>Do PostgreSQL&rsquo;s estimates match reality?</h2>
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
        <h2>What we found in your database</h2>
        <SchemaPanel schema={schema} />
      </section>
    </>
  );
}
