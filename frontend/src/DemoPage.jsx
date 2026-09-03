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
      // The structured explanation when the backend produced one (a 400 about
      // the SQL itself), the bare message otherwise. `SqlErrorPanel` renders
      // both, so nothing here has to decide which case it is.
      setError(err.sqlError ?? err.message);
    } finally {
      setLoading(false);
    }
  }

  const overall = trend?.overall;

  return (
    <>
      {/* A plain intro line, not a second site header. Reusing `app-header`
          here gave a one-sentence paragraph the header's border, padding and
          bottom margin -- a stray rule across the page and a large gap under
          it. */}
      <p className="page-intro">
        Paste a query and see how PostgreSQL&rsquo;s plan compares to the one the{" "}
        {overall?.selector_mode ?? "…"} optimizer picks.
      </p>

      {/* Grouping the fourteen panels was an improvement over a flat list, but
          it left every one of them open: 4,200 pixels of prose and charts under
          a query box, with no way to tell what to look at.

          The page does one thing now. Paste a query, get an answer, see what
          would fix it properly. Everything else is diagnostics -- true, worth
          having, and not what anyone opens this to read -- so it is folded away
          behind two toggles that say what is inside them. */}
      <section className="card">
        <h2>Query</h2>
        <QueryForm
          onAnalyze={handleAnalyze}
          loading={loading}
          error={error}
          onUseSuggestion={(sql) => {
            setError(null);
            handleAnalyze(sql);
          }}
        />
      </section>

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

      {/* A hint fixes one query; these fix the reason it needed one. Kept open
          and next to the result, because it is the other half of the answer --
          not a diagnostic. */}
      <section className="card">
        <h2>Fix the cause, not the symptom</h2>
        <Recommendations queryRecs={result?.recommendations} schemaRecs={schemaRecs} />
      </section>

      {/* Everything below is true and worth having, and none of it is why
          anyone opened the page. Two toggles, closed. */}
      <h2 className="group-heading">Going deeper</h2>

      <Collapsible
        title="Is it working?"
        hint="time saved, decision quality, and how good PostgreSQL's estimates are"
      >
        <p className="decision-caution" style={{ marginTop: 0 }}>
          Every figure is a matched pair &mdash; PostgreSQL&rsquo;s plan and the served plan for
          the same query, in the same run. One run settles very little;{" "}
          <em>Settle a comparison</em> below is the tool for that.
        </p>
        {trendError && <div className="error-banner">{trendError}</div>}
        <div className="card-grid">
          {/* Spans the grid: it carries a table of every run, which a 340px
              track squeezes into a horizontal scrollbar with clipped headers. */}
          <section className="card card-wide">
            <h2>How much time this is saving</h2>
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
            then the run sequence above is the honest time axis. */}
        {trend?.by_day?.length > 1 && (
          <section className="card">
            <h2>Day by day</h2>
            <TrendChart byDay={trend.by_day} />
          </section>
        )}
      </Collapsible>

      <Collapsible
        title="The database, and the model"
        hint="schema, health checks, model status, and a paired A/B"
      >
        <section className="card">
          <h2>Database health</h2>
          <DatabaseAdvisor
            findings={dbFindings}
            onRefresh={loadDatabaseAdvisor}
            busy={dbBusy}
            deep={dbDeep}
            onDeepChange={setDbDeep}
          />
        </section>

        <section className="card">
          <h2>What we found in your database</h2>
          <SchemaPanel schema={schema} />
        </section>

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

        <section className="card">
          <h2>Settle a comparison</h2>
          <ExperimentPanel status={experiment} onStart={handleStartExperiment} busy={modelBusy} />
        </section>

        {result && (
          <section className="card">
            <h2>Production path</h2>
            <ProductionRun
              result={production}
              onRun={handleProductionRun}
              busy={productionBusy}
            />
          </section>
        )}
      </Collapsible>
    </>
  );
}
