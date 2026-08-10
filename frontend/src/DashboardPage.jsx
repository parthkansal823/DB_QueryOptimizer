import { useCallback, useEffect, useState } from "react";
import {
  activateDatabase,
  addDatabase,
  fetchDatabases,
  fetchSettings,
  removeDatabase,
  resetSettings,
  testDatabase,
  updateSettings,
} from "./api";

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

export default function SettingsPage() {
  const [settings, setSettings] = useState(null);
  const [databases, setDatabases] = useState(null);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);

  const loadDatabases = useCallback(async () => {
    setDatabases(await fetchDatabases());
  }, []);

  useEffect(() => {
    fetchSettings().then(setSettings).catch((err) => setError(err.message));
    loadDatabases().catch((err) => setError(err.message));
  }, [loadDatabases]);

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
    return <section className="card empty-state">Could not load settings: {error}</section>;
  }
  if (!settings) {
    return <section className="card empty-state">Loading settings…</section>;
  }

  return (
    <>
      <header className="app-header">
        <p>
          Change how the optimizer behaves without restarting it. Every setting applies to the
          next query and is remembered across restarts.
        </p>
      </header>

      {error && <p className="decision-caution">{error}</p>}
      {saved && !error && <p className="probe-ok">Saved and applied.</p>}

      <DatabasePanel databases={databases} onChanged={loadDatabases} onError={setError} />

      {settings.groups.map((group) => (
        <section className="card" key={group}>
          <h2>{group}</h2>
          {settings.fields
            .filter((field) => field.group === group)
            .map((field) => (
              <Field
                key={field.name}
                field={field}
                value={settings.values[field.name]}
                onChange={handleChange}
                disabled={busy}
              />
            ))}
        </section>
      ))}

      <section className="card">
        <button type="button" onClick={handleReset} disabled={busy}>
          Reset to defaults
        </button>
      </section>
    </>
  );
}
