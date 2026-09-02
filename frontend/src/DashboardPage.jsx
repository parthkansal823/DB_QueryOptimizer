import { useCallback, useEffect, useState } from "react";
import {
  activateDatabase,
  addDatabase,
  fetchDatabases,
  fetchSavedQueries,
  fetchSettings,
  removeDatabase,
  resetSettings,
  testDatabase,
  updateSettings,
} from "./api";
import Collapsible from "./components/Collapsible";
import HowToUse from "./components/HowToUse";
import QueryManager from "./components/QueryManager";
import TrainingPanel from "./components/TrainingPanel";

/**
 * One setting, rendered from the metadata the backend declares for it.
 *
 * Nothing here knows what any particular field means -- adding a field to
 * `app/settings.py` makes it appear with no frontend change, and removing one
 * cannot leave a control here pointing at something that no longer exists.
 */
function Field({ field, value, onChange, disabled }) {
  const id = `setting-${field.name}`;

  return (
    <div className="setting-row">
      <label htmlFor={id}>{field.label}</label>

      {field.type === "bool" && (
        <input
          id={id}
          type="checkbox"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(e) => onChange(field.name, e.target.checked)}
        />
      )}

      {field.type === "enum" && (
        <select
          id={id}
          value={value ?? ""}
          disabled={disabled}
          onChange={(e) => onChange(field.name, e.target.value)}
        >
          {field.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      )}

      {(field.type === "float" || field.type === "int") && (
        <input
          id={id}
          type="number"
          value={value ?? ""}
          min={field.min}
          max={field.max}
          step={field.step}
          disabled={disabled}
          onChange={(e) => onChange(field.name, e.target.value)}
        />
      )}

      <p className="setting-help">{field.help}</p>
    </div>
  );
}

function DatabasePanel({ databases, onChanged, onError }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [probe, setProbe] = useState(null);
  const [busy, setBusy] = useState(false);

  async function run(action) {
    setBusy(true);
    onError(null);
    try {
      await action();
      await onChanged();
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleTest() {
    setBusy(true);
    onError(null);
    try {
      setProbe(await testDatabase(url));
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const fit = databases?.model_fit;

  return (
    <section className="card">
      <h2>Databases</h2>
      <p className="setting-help">
        Point the optimizer at another PostgreSQL database. Connections are saved by name;
        making one active repoints the connection pool immediately.
      </p>

      <p className="active-db">
        Active: <code>{databases?.active_url ?? "…"}</code>
      </p>

      {/* A trained model carries the feature columns of the schema it was
          built on, so this is the difference between a dashboard reporting
          learned decisions and one reporting confident nonsense. */}
      {fit?.model_trained && fit.matches === false && (
        <p className="decision-caution">
          The served model was trained on a different schema — it is missing{" "}
          <strong>{fit.missing_tables.join(", ")}</strong>. Its predictions do not describe this
          database. Run <code>python -m app.onboard</code> against it before trusting them.
        </p>
      )}

      <ul className="db-list">
        {(databases?.profiles ?? []).map((profile) => (
          <li key={profile.name} className={profile.active ? "db-item active" : "db-item"}>
            <span className="db-name">
              {profile.active ? "● " : "○ "}
              {profile.name}
            </span>
            <code>{profile.url}</code>
            <span className="db-actions">
              {!profile.active && (
                <button
                  type="button"
                  disabled={busy || !databases?.allow_runtime_change}
                  onClick={() => run(() => activateDatabase(profile.name))}
                >
                  Make active
                </button>
              )}
              <button
                type="button"
                disabled={busy || profile.active}
                onClick={() => run(() => removeDatabase(profile.name))}
              >
                Remove
              </button>
            </span>
          </li>
        ))}
        {(databases?.profiles ?? []).length === 0 && (
          <li className="empty-state">No saved connections yet.</li>
        )}
      </ul>

      {databases && !databases.allow_runtime_change && (
        <p className="decision-caution">
          Runtime switching is disabled (<code>ALLOW_RUNTIME_DB_CHANGE=0</code>).
        </p>
      )}

      <div className="db-add">
        <input
          aria-label="Connection name"
          placeholder="name (e.g. imdb-job)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          aria-label="Connection URL"
          placeholder="postgresql://user:password@host:5432/dbname"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button type="button" onClick={handleTest} disabled={busy || !url}>
          Test
        </button>
        <button
          type="button"
          onClick={() => run(() => addDatabase(name, url).then(() => { setName(""); setUrl(""); setProbe(null); }))}
          disabled={busy || !url || !name}
        >
          Add
        </button>
      </div>

      {probe && (
        <p className={probe.ok ? "probe-ok" : "probe-fail"}>
          {probe.ok ? (
            <>
              Connected: {probe.server_version} — {probe.n_tables} tables
              {!probe.pg_hint_plan && (
                <>
                  {" "}
                  <strong>
                    pg_hint_plan is not installed here, so every hint would be silently ignored
                    and no candidate could differ from the native plan.
                  </strong>
                </>
              )}
            </>
          ) : (
            <>Could not connect: {probe.error}</>
          )}
        </p>
      )}
    </section>
  );
}

export default function DashboardPage() {
  const [settings, setSettings] = useState(null);
  const [databases, setDatabases] = useState(null);
  const [savedQueries, setSavedQueries] = useState(null);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);

  const loadDatabases = useCallback(async () => {
    setDatabases(await fetchDatabases());
  }, []);

  const loadQueries = useCallback(async () => {
    setSavedQueries(await fetchSavedQueries());
  }, []);

  // A finished training run changes the served model and, if the guard
  // re-evaluated, which queries are blocked -- so both panels are re-read
  // rather than left showing pre-training state.
  const handleTrained = useCallback(() => {
    loadDatabases().catch(() => {});
  }, [loadDatabases]);

  useEffect(() => {
    fetchSettings().then(setSettings).catch((err) => setError(err.message));
    loadDatabases().catch((err) => setError(err.message));
    loadQueries().catch((err) => setError(err.message));
  }, [loadDatabases, loadQueries]);

  async function handleChange(name, value) {
    // Applied on change rather than behind a Save button: every one of these
    // takes effect on the next query, and the point of putting them here was
    // to make "change it and watch what happens" a single action.
    setBusy(true);
    setError(null);
    try {
      const result = await updateSettings({ [name]: value });
      setSettings((prev) => ({ ...prev, values: result.values }));
      setSaved(true);
    } catch (err) {
      setError(err.message);
      // Re-read rather than keeping the rejected value on screen, so the
      // control always shows what the server actually holds.
      fetchSettings().then(setSettings).catch(() => {});
    } finally {
      setBusy(false);
    }
  }

  async function handleReset() {
    setBusy(true);
    try {
      const result = await resetSettings();
      setSettings((prev) => ({ ...prev, values: result.values }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (error && !settings) {
    return <section className="card empty-state">Could not load the dashboard: {error}</section>;
  }
  if (!settings) {
    return <section className="card empty-state">Loading dashboard…</section>;
  }

  return (
    <>
      {/* A plain intro line, not a second site header -- reusing `app-header`
          gave one sentence the header's border, padding and bottom margin. */}
      <p className="page-intro">
        Connect a database, save the queries you care about, train on them, and tune how the
        optimizer behaves — all without touching the backend. Every setting applies to the next
        query and is remembered across restarts.
      </p>

      {error && <p className="decision-caution">{error}</p>}
      {saved && !error && <p className="probe-ok">Saved and applied.</p>}

      {/* The instructions live here, not on the demo page. The demo exists to
          show a result; instructions above the result push it off the screen,
          and this is the page you are on when you are setting things up. */}
      <HowToUse />

      <DatabasePanel databases={databases} onChanged={loadDatabases} onError={setError} />

      <QueryManager
        queries={savedQueries?.queries}
        onChanged={loadQueries}
        onError={setError}
      />

      <TrainingPanel
        nQueries={savedQueries?.queries?.length ?? 0}
        builtinCount={savedQueries?.builtin_count}
        onFinished={handleTrained}
        onError={setError}
      />

      {/* Settings are folded away, one toggle per group.
          There are now more than thirty of them across five groups -- every
          hint family added from the pg_hint_plan reference brought one -- and
          rendering them all open made this page five thousand pixels tall. The
          three panels above are what someone comes here to do; the settings are
          what they change once and then leave alone.

          Each toggle says how many settings are inside and how many are no
          longer at their default, so a changed setting is findable without
          opening all five. */}
      <h2 className="group-heading">Settings</h2>

      {settings.groups.map((group) => {
        const fields = settings.fields.filter((field) => field.group === group);
        const changed = fields.filter(
          (field) => settings.values[field.name] !== field.default,
        ).length;
        return (
          <Collapsible
            key={group}
            title={group}
            hint={
              changed
                ? `${fields.length} settings, ${changed} changed`
                : `${fields.length} settings, all default`
            }
          >
            {fields.map((field) => (
              <Field
                key={field.name}
                field={field}
                value={settings.values[field.name]}
                onChange={handleChange}
                disabled={busy}
              />
            ))}
          </Collapsible>
        );
      })}

      <p>
        <button
          type="button"
          className="button-secondary"
          onClick={handleReset}
          disabled={busy}
        >
          Reset all settings to defaults
        </button>
      </p>
    </>
  );
}
