import './App.css';
import { Routes, Route, BrowserRouter } from 'react-router-dom';
import FileProblem from './pages/fileProblem';
import DistanceEstimator from './pages/cameraDistance';


// const Home = () => {
//   return (
//     <div>

//     </div>
//   )
// }

function App() {

  return (
    <BrowserRouter>
      <Routes>
        <Route index element={<DistanceEstimator />} />
        <Route path="/file-problem" element={<FileProblem />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;