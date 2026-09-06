import { useEffect, useState } from "react";

import { fetchSettings } from "../api";

// Settling a comparison, rather than reading one number off one run.
//
// Every other panel here reports a single sweep. `app.experiment` interleaves
// two arms, drops a pair when either half fails, and reports a bootstrap
// interval plus a sign test -- and calls the result *unresolved* when the
// interval spans zero, however good the mean looks. That was CLI-only, so the
// rigorous comparison was the inconvenient one; docs/WRITEUP.md 2.4 records
// what that cost, a headline result with a sample size of three.

function Verdict({ result }) {
  if (!result) return null;

  const unresolved = String(result.verdict ?? "").toLowerCase().includes("unresolved");
  const ci = result.confidence_interval ?? result.ci;

  return (
    <div className="decision-note">
      <dl className="decision-facts">
        <dt>Verdict</dt>
        <dd>
          <strong>{result.verdict ?? "—"}</strong>
        </dd>
        {result.mean_difference != null && (
          <>
            <dt>Mean difference</dt>
            <dd>{result.mean_difference.toFixed(2)} pp of oracle headroom</dd>
          </>
        )}
        {Array.isArray(ci) && ci.length === 2 && (
          <>
            <dt>95% interval</dt>
            <dd>
              [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]
            </dd>
          </>
        )}
        {result.sign_test_p_value != null && (
          <>
            <dt>Sign test</dt>
            <dd>p = {result.sign_test_p_value.toFixed(3)}</dd>
          </>
        )}
        {result.n_pairs != null && (
          <>
            <dt>Paired runs</dt>
            <dd>{result.n_pairs}</dd>
          </>
        )}
      </dl>
      {unresolved && (
        <p className="decision-caution">
          The interval spans zero, so this comparison has not resolved. More paired runs
          would narrow it; a bigger mean would not.
        </p>
      )}
    </div>
  );
}

// Fallback only. The real list comes from `/settings`, because the backend is
// where policies are added and this component is the eighth place the list used
// to be spelled out -- after the autotune sweep, the benchmark's --policy, the
// calibration replay, the promotion yardstick, the settings API, the experiment
// runner and the training evaluation, all of which now read one constant
// (`learned.RANKER_POLICIES`).
//
// `borda` was added to all seven and not to this file, so the dashboard offered
// an A/B over four policies while the server ran five. A panel that cannot
// select the policy you want to compare is a panel that quietly decides the
// comparison for you.
const FALLBACK_POLICIES = ["greedy", "thompson", "risk_averse", "pairwise_rank"];

export default function ExperimentPanel({ status, onStart, busy }) {
  const [comparison, setComparison] = useState("guard");
  const [armA, setArmA] = useState("risk_averse");
  const [armB, setArmB] = useState("pairwise_rank");
  const [policy, setPolicy] = useState("pairwise_rank");
  const [runs, setRuns] = useState(10);
  const [error, setError] = useState(null);
  const [policies, setPolicies] = useState(FALLBACK_POLICIES);

  // The settings endpoint already declares the options for `selection_policy`,
  // and the settings page renders them without knowing what they mean. This
  // borrows the same list rather than keeping a second one.
  useEffect(() => {
    let cancelled = false;
    fetchSettings()
      .then((settings) => {
        const field = settings?.fields?.find((f) => f.name === "selection_policy");
        if (!cancelled && field?.options?.length) setPolicies(field.options);
      })
      .catch(() => {
        // A panel that cannot reach the API still has to render; the fallback
        // is stale rather than wrong.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const running = status?.is_running;

  async function start() {
    setError(null);
    try {
      await onStart({ comparison, policy, arm_a: armA, arm_b: armB, runs: Number(runs) });
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div>
      <div className="settings-grid">
        <label>
          Compare
          <select
            value={comparison}
            onChange={(e) => setComparison(e.target.value)}
            disabled={running}
          >
            <option value="guard">Regression guard on vs off</option>
            <option value="policy">Two selection policies</option>
          </select>
        </label>

        {comparison === "guard" ? (
          <label>
            Policy
            <select value={policy} onChange={(e) => setPolicy(e.target.value)} disabled={running}>
              {policies.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <>
            <label>
              Arm A
              <select value={armA} onChange={(e) => setArmA(e.target.value)} disabled={running}>
                {policies.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Arm B
              <select value={armB} onChange={(e) => setArmB(e.target.value)} disabled={running}>
                {policies.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}

        <label>
          Paired runs
          <input
            type="number"
            min="1"
            max="50"
            value={runs}
            onChange={(e) => setRuns(e.target.value)}
            disabled={running}
          />
        </label>
      </div>

      <p className="decision-caution">
        Each paired run is two full benchmark sweeps, so this takes minutes to hours. It
        runs in the background; this panel can be left and come back to.
      </p>

      <button type="button" onClick={start} disabled={running || busy}>
        {running ? `Running ${status.label}…` : "Run comparison"}
      </button>

      {error && <p className="decision-veto">{error}</p>}
      {status?.error && <p className="decision-veto">Failed: {status.error}</p>}

      {status?.label && (
        <p style={{ marginTop: 12 }}>
          <strong>{status.label}</strong>
          {status.elapsed_seconds != null && ` — ${status.elapsed_seconds}s`}
        </p>
      )}

      <Verdict result={status?.result} />
    </div>
  );
}
