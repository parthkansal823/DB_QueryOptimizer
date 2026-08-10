import { NavLink, Route, HashRouter as Router, Routes } from "react-router-dom";
import "./App.css";
import Dashboard from "./Dashboard";
import SettingsPage from "./SettingsPage";

// HashRouter rather than BrowserRouter: the dashboard is served by `vite
// preview` and by a plain static host in the Docker image, neither of which
// rewrites unknown paths to index.html. With browser history routing, opening
// /settings directly or refreshing on it would 404 -- the classic SPA deploy
// footgun, and not worth a server config for a two-page tool.
export default function App() {
  return (
    <Router>
      <header className="app-header app-nav">
        <h1>Learned Query Optimizer</h1>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Dashboard
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Settings
          </NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/settings" element={<SettingsPage />} />
        {/* An unknown hash lands on the dashboard rather than a blank page. */}
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}
