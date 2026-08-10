import { useEffect, useRef, useState } from "react";
import { fetchTrainStatus, startTraining, stopTraining } from "../api";

const POLL_MS = 1500;

/**
 * Start a training run and watch it.
 *
 * Collection executes every candidate plan for every query, so a run is
 * minutes long -- which is why this polls and shows progress rather than
 * blocking on a request that would time out long before finishing.
 */
export default function TrainingPanel({ nQueries, builtinCount, onFinished, onError }) {
  const [status, setStatus] = useState(null);
  const [reps, setReps] = useState(3);
  const [promote, setPromote] = useState(true);
  const [includeBuiltin, setIncludeBuiltin] = useState(false);
  const wasRunning = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const next = await fetchTrainStatus();
        if (cancelled) return;
        setStatus(next);

        // Only refresh the rest of the page on the running -> finished edge.
        // Refreshing on every poll would reload the model status once a second
        // for the whole run.
        if (wasRunning.current && !next.is_running) onFinished?.();
        wasRunning.current = next.is_running;
      } catch {
        // A poll that fails is not worth surfacing -- the next one usually
        // succeeds, and a backend restart mid-run should not paint an error.
      }
    }

    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [onFinished]);

  async function handleStart() {
    onError(null);
    try {
      setStatus(
        await startTraining({
          reps,
          promote,
          include_builtin: includeBuiltin,
          include_join_methods: true,
        }),
      );
      wasRunning.current = true;
    } catch (err) {
      onError(err.message);
    }
  }

  async function handleStop() {
    try {
      setStatus(await stopTraining());
    } catch (err) {
      onError(err.message);
    }
  }

  const running = status?.is_running;
  const total = (includeBuiltin ? builtinCount ?? 0 : 0) + (nQueries ?? 0);
  const percent = status?.total ? Math.round((status.done / status.total) * 100) : 0;

  return (
    <section className="card">
      <h2>Train the model</h2>
      <p className="setting-help">
        Runs every candidate plan for every saved query to gather labelled timings, then fits
        the model on them. This takes minutes, not seconds — each query is executed once per
        candidate plan, several times over.
      </p>

      <div className="train-controls">
        <label>
          Repetitions per plan
          <input
            type="number"
            min={1}
            max={10}
            value={reps}
            disabled={running}
            onChange={(e) => setReps(Number(e.target.value))}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={promote}
            disabled={running}
            onChange={(e) => setPromote(e.target.checked)}
          />
          Promote if it beats the current model
        </label>
        <label>
          <input
            type="checkbox"
            checked={includeBuiltin}
            disabled={running}
            onChange={(e) => setIncludeBuiltin(e.target.checked)}
          />
          Also include the {builtinCount ?? 0} built-in benchmark queries
        </label>
      </div>

      {/* More repetitions is the direct lever on label quality, and the noise
          floor is the reason it matters -- a single timing cannot tell a real
          20% win from the same plan run at a luckier moment. */}
      <p className="setting-help">
        Higher repetitions give better labels: the median of several runs is used, which is
        what keeps measurement noise out of the training target.
      </p>

      <div className="train-actions">
        <button type="button" onClick={handleStart} disabled={running || total === 0}>
          {running ? "Training…" : `Start training on ${total} quer${total === 1 ? "y" : "ies"}`}
        </button>
        {running && (
          <button type="button" onClick={handleStop}>
            Stop
          </button>
        )}
      </div>

      {total === 0 && (
        <p className="setting-help">
          Save a query first, or tick the built-in benchmark workload.
        </p>
      )}

      {status && status.status !== "idle" && (
        <div className="train-status">
          <div className="train-progress">
            <div className="train-bar" style={{ width: `${percent}%` }} />
          </div>
          <p>
            <strong>{status.status}</strong>
            {status.stage && ` — ${status.stage}`}
            {status.total > 0 && ` — ${status.done}/${status.total} queries`}
            {status.rows_collected > 0 && `, ${status.rows_collected} rows`}
            {status.elapsed_seconds != null && ` (${status.elapsed_seconds}s)`}
          </p>

          {status.error && <p className="probe-fail">{status.error}</p>}

          {status.status === "stopped" && (
            <p className="setting-help">
              Stopped. The rows collected so far are kept — starting again adds to them
              rather than beginning from scratch.
            </p>
          )}

          {status.status === "done" && status.result && (
            <p className="probe-ok">
              Finished: {status.result.action ?? "trained"}
              {status.result.reason ? ` — ${status.result.reason}` : ""}
            </p>
          )}

          {status.log?.length > 0 && (
            <pre className="train-log">{status.log.slice(-12).join("\n")}</pre>
          )}
        </div>
      )}
    </section>
  );
}
