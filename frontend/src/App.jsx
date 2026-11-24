import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar.jsx';
import LandingPage from './pages/LandingPage.jsx';
import Dashboard from './pages/Dashboard.jsx';
import EnhancementPage from './pages/EnhancementPage.jsx';
import ComparisonPage from './pages/ComparisonPage.jsx';
import DetectionPage from './pages/DetectionPage.jsx';
import AnalyticsPage from './pages/AnalyticsPage.jsx';
import DatasetPage from './pages/DatasetPage.jsx';
import AboutPage from './pages/AboutPage.jsx';
import './styles/App.css';

function App() {
  return (
    <div className="app-container">
      <Routes>
        <Route path="/landing" element={<LandingPage />} />
        <Route
          path="/*"
          element={
            <div className="app-layout">
              <Sidebar />
              <main className="main-content">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/enhance" element={<EnhancementPage />} />
                  <Route path="/compare" element={<ComparisonPage />} />
                  <Route path="/detect" element={<DetectionPage />} />
                  <Route path="/analytics" element={<AnalyticsPage />} />
                  <Route path="/datasets" element={<DatasetPage />} />
                  <Route path="/about" element={<AboutPage />} />
                </Routes>
              </main>
            </div>
          }
        />
      </Routes>
    </div>
  );
}

export default App;
