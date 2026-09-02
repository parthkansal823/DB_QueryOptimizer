import { useEffect, useState } from "react";
import {
  analyzeQuery,
  fetchAdvisor,
  fetchCostModel,
  fetchDatabaseAdvisor,
  fetchExperimentStatus,
  fetchModelDiagnostics,
  fetchModelStatus,
  fetchSchema,
  startExperiment,
  fetchTrend,
  optimizeQuery,
  triggerRetrain,
  triggerRollback,
} from "./api";
import CostModelChart from "./components/CostModelChart";
import CumulativeChart from "./components/CumulativeChart";
import DecisionQuality from "./components/DecisionQuality";
import LatencyChart from "./components/LatencyChart";
import DatabaseAdvisor from "./components/DatabaseAdvisor";
import ExperimentPanel from "./components/ExperimentPanel";
import Collapsible from "./components/Collapsible";
import HowToUse from "./components/HowToUse";
import ModelHealth from "./components/ModelHealth";
import OptimizedQuery from "./components/OptimizedQuery";
import Recommendations from "./components/Recommendations";
import ProductionRun from "./components/ProductionRun";
import SchemaPanel from "./components/SchemaPanel";
import PlanComparison from "./components/PlanComparison";
import QueryForm from "./components/QueryForm";
import ServedVsNative from "./components/ServedVsNative";
import TrendChart from "./components/TrendChart";

export default function DemoPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trend, setTrend] = useState(null);
  const [trendError, setTrendError] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [modelDiagnostics, setModelDiagnostics] = useState(null);
  const [experiment, setExperiment] = useState(null);
  const [dbFindings, setDbFindings] = useState(null);
  const [dbBusy, setDbBusy] = useState(false);
  const [dbDeep, setDbDeep] = useState(false);
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

  async function loadDatabaseAdvisor() {
    setDbBusy(true);
    try {
      setDbFindings(await fetchDatabaseAdvisor(dbDeep));
    } catch {
      setDbFindings(null);
    } finally {
      setDbBusy(false);
    }
  }

  async function loadExperiment() {
    try {
      setExperiment(await fetchExperimentStatus());
    } catch {
      setExperiment(null);
    }
  }

  async function handleStartExperiment(body) {
    setExperiment(await startExperiment(body));
    // A comparison is minutes to hours, so poll rather than await it.
    const timer = setInterval(async () => {
      const next = await fetchExperimentStatus();
      setExperiment(next);
      if (!next.is_running) clearInterval(timer);
    }, 5000);
  }

  async function loadModelStatus() {
    try {
      setModelStatus(await fetchModelStatus());
    } catch {
      setModelStatus(null);
    }
    // Fetched separately so a backend too old to serve it still renders the
    // rest of the panel rather than blanking it.
    try {
      setModelDiagnostics(await fetchModelDiagnostics());
    } catch {
      setModelDiagnostics(null);
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
    loadExperiment();
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

      {/* The page used to be fourteen cards in a flat list, which is readable
          only if you already know which ones answer which question. Grouping
          them says what each run of panels is for: this query, then this
          database, then whether any of it is working, then the model itself. */}
      <h2 className="group-heading">This query</h2>

      <section className="card">
        <h2>Query</h2>
        <QueryForm onAnalyze={handleAnalyze} loading={loading} error={error} />
      </section>

      {/* Below the query box, not above it. The instructions are worth having
          and they are not what anyone came here to read first -- opening on a
          screenful of them pushes the one control that does anything below the
          fold. */}
      <HowToUse />

      {result && (
        <>
          {/* The success styling is conditional on there being a success. A
              green-tinted card reading "nothing beat PostgreSQL" says two
              opposite things at once, and the colour is what a reader takes
              in first. */}
          <section className={`card${result.best_measured ? " card-hero" : ""}`}>
            <h2>Faster version of your query</h2>
            <OptimizedQuery
              best={result.best_measured}
              decision={result.decision}
              candidatesTried={result.candidates?.length}
              incomingHints={result.incoming_hints}
            />
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

      {result && (
        <section className="card">
          <h2>Production path</h2>
          <ProductionRun result={production} onRun={handleProductionRun} busy={productionBusy} />
        </section>
      )}

      <h2 className="group-heading">This database</h2>
      <p className="group-blurb">
        Findings about the database rather than one query&rsquo;s plan. Each is fixed once and
        then helps every query, including ones this system has never seen. Nothing here is
        applied &mdash; every finding shows the statement that would apply it.
      </p>

      <Collapsible title="Database health" hint="correlated columns, indexes, settings">
        <DatabaseAdvisor
          findings={dbFindings}
          onRefresh={loadDatabaseAdvisor}
          busy={dbBusy}
          deep={dbDeep}
          onDeepChange={setDbDeep}
        />
      </Collapsible>

      <section className="card">
        <h2>Database fixes &mdash; treat the cause, not the symptom</h2>
        <Recommendations queryRecs={result?.recommendations} schemaRecs={schemaRecs} />
      </section>

      <Collapsible title="What we found in your database" hint="tables and join edges">
        <SchemaPanel schema={schema} />
      </Collapsible>

      <h2 className="group-heading">Is it working?</h2>
      <p className="group-blurb">
        Every figure here is a matched pair &mdash; PostgreSQL&rsquo;s plan and the served plan
        for the same query, measured in the same run. A single run settles very little on this
        database; <em>Settle a comparison</em> below is the tool for that.
      </p>

      <div className="card-grid">
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
      </div>

      {/* A per-day rollup needs more than one day to mean anything. Until
          then the run sequence above is the honest time axis, and a two-point
          line chart would just be decoration. */}
      {trend?.by_day?.length > 1 && (
        <section className="card">
          <h2>Day by day</h2>
          <TrendChart byDay={trend.by_day} />
        </section>
      )}

      <h2 className="group-heading">The model itself</h2>
      <p className="group-blurb">
        Whether the model can be believed, and how to settle a question about it properly
        rather than by reading three numbers off one run.
      </p>

      <section className="card">
        <h2>Model health</h2>
        <ModelHealth
          status={modelStatus}
          diagnostics={modelDiagnostics}
          busy={modelBusy}
          onRetrain={() => runModelAction(triggerRetrain)}
          onRollback={() => runModelAction(triggerRollback)}
        />
      </section>

      <Collapsible title="Settle a comparison" hint="paired A/B with a confidence interval">
        <ExperimentPanel status={experiment} onStart={handleStartExperiment} busy={modelBusy} />
      </Collapsible>
    </>
  );
}
