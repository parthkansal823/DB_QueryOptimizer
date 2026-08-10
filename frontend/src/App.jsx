import { NavLink, Route, HashRouter as Router, Routes } from "react-router-dom";
import "./App.css";
import DashboardPage from "./DashboardPage";
import DemoPage from "./DemoPage";

// HashRouter rather than BrowserRouter: the dashboard is served by `vite
// preview` and by a plain static host in the Docker image, neither of which
// rewrites unknown paths to index.html. With browser history routing, opening
// /demo directly or refreshing on it would 404 -- the classic SPA deploy
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
          <NavLink to="/demo" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Demo
          </NavLink>
        </nav>
      </header>

      <Routes>
        {/* The control centre: connect a database, save your own queries,
            train on them, and tune how the optimizer behaves. */}
        <Route path="/" element={<DashboardPage />} />
        {/* The presentation piece, unchanged: paste a query and watch the
            optimizer work through it. */}
        <Route path="/demo" element={<DemoPage />} />
        <Route path="*" element={<DashboardPage />} />
      </Routes>
    </Router>
  );
}
