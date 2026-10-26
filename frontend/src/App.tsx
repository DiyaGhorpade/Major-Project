import React, { useState, useEffect } from 'react';
import { AppShell } from './components/layout/AppShell';
import { NavRoute } from './components/layout/Sidebar';

// Pages
import { OverviewPage } from './pages/OverviewPage';
import { LiveMonitoringPage } from './pages/LiveMonitoringPage';
import { PlantsPage } from './pages/PlantsPage';
import { PlantDetailPage } from './pages/PlantDetailPage';
import { AiInsightsPage } from './pages/AiInsightsPage';
import { AnomaliesPage } from './pages/AnomaliesPage';
import { parseAnomalyPlantId } from './lib/anomalyFilters';
import { ShapAnalysisPage } from './pages/ShapAnalysisPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { RecommendationsPage } from './pages/RecommendationsPage';
import { DataLogsPage } from './pages/DataLogsPage';
import { DevicesNetworkPage } from './pages/DevicesNetworkPage';
import { SessionsPage } from './pages/SessionsPage';
import { SettingsPage } from './pages/SettingsPage';

export function App() {
  const [currentRoute, setCurrentRoute] = useState<NavRoute>('overview');
  const [activePlantId, setActivePlantId] = useState<number | string>(1);
  // Plant the Anomalies page opens filtered to (`#/anomalies/<id>`); null shows all plants.
  const [anomalyPlantId, setAnomalyPlantId] = useState<string | null>(null);

  // Sync route with URL hash for browser back/forward and deep linking
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#/', '').replace('#', '');
      if (hash.startsWith('plant/')) {
        const id = hash.replace('plant/', '');
        setActivePlantId(id);
        setCurrentRoute('plant-detail');
      } else if (hash.startsWith('shap/')) {
        const id = hash.replace('shap/', '');
        setActivePlantId(id);
        setCurrentRoute('shap-analysis');
      } else if (hash === 'anomalies' || hash.startsWith('anomalies/')) {
        setAnomalyPlantId(parseAnomalyPlantId(hash.replace('anomalies', '').replace('/', '')));
        setCurrentRoute('anomalies');
      } else if (hash) {
        setCurrentRoute(hash as NavRoute);
      }
    };

    if (window.location.hash) {
      handleHashChange();
    }

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  
  const handleNavigate = (route: NavRoute, plantId?: number | string) => {
    if (route === 'anomalies') {
      setAnomalyPlantId(plantId === undefined ? null : String(plantId));
      window.location.hash = plantId === undefined ? '#/anomalies' : `#/anomalies/${plantId}`;
      setCurrentRoute(route);
      return;
    }
    if (plantId !== undefined) {
      setActivePlantId(plantId);

      if (route === 'plant-detail') {
        window.location.hash = `#/plant/${plantId}`;
      } else if (route === 'shap-analysis') {
        window.location.hash = `#/shap/${plantId}`;
      } else {
        window.location.hash = `#/${route}`;
      }
    } else {
      window.location.hash = `#/${route}`;
    }

    setCurrentRoute(route);
  };


  const renderActivePage = () => {
    switch (currentRoute) {
      case 'overview':
        return <OverviewPage onNavigate={handleNavigate} />;
      case 'live-monitoring':
        return <LiveMonitoringPage onNavigate={handleNavigate} />;
      case 'plants':
        return <PlantsPage onNavigate={handleNavigate} />;
      case 'plant-detail':
        return <PlantDetailPage plantId={activePlantId} onNavigate={handleNavigate} />;
      case 'ai-insights':
        return <AiInsightsPage onNavigate={handleNavigate} />;
      case 'anomalies':
        return <AnomaliesPage key={anomalyPlantId ?? 'all'} initialPlantId={anomalyPlantId} onNavigate={handleNavigate} />;
      case 'shap-analysis':
        return <ShapAnalysisPage initialPlantId={activePlantId} onNavigate={handleNavigate} />;
      case 'model-performance':
        return <ModelPerformancePage />;
      case 'recommendations':
        return <RecommendationsPage onNavigate={handleNavigate} />;
      case 'data-logs':
        return <DataLogsPage />;
      case 'devices-network':
        return <DevicesNetworkPage />;
      case 'sessions':
        return <SessionsPage onNavigate={handleNavigate} />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <OverviewPage onNavigate={handleNavigate} />;
    }
  };

  return (
    <AppShell
      currentRoute={currentRoute}
      activePlantId={currentRoute === 'plant-detail' ? activePlantId : null}
      onNavigate={handleNavigate}
    >
      {renderActivePage()}
    </AppShell>
  );
}

export default App;
