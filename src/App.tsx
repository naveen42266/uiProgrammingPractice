import './App.css';
import { Routes, Route, BrowserRouter, Link } from 'react-router-dom';
import FileProblem from './pages/fileProblem';
import DistanceEstimator from './pages/cameraDistance';

function App() {
  return (
    <BrowserRouter>
      <nav className="bg-slate-900 border-b border-slate-800 px-4 py-2 flex items-center justify-between text-xs font-semibold">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-emerald-400 hover:text-emerald-300 flex items-center gap-1.5">
            AI Distance Meter
          </Link>
          <Link to="/file-problem" className="text-slate-400 hover:text-slate-200">
            File Problem
          </Link>
        </div>
      </nav>

      <Routes>
        <Route index element={<DistanceEstimator />} />
        <Route path="/distance" element={<DistanceEstimator />} />
        <Route path="/file-problem" element={<FileProblem />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;