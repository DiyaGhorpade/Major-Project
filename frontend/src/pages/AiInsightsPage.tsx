import React, { useState, useEffect } from 'react';
import {
  BrainCircuit,
  Cpu,
  PieChart,
  Target,
  Sparkles,
  ArrowRight,
  ShieldAlert,
  Database,
  Layers,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Plant, ModelMetrics } from '../types';
import { MetricCard } from '../components/shared/MetricCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { EmptyState } from '../components/shared/EmptyState';
import { NavRoute } from '../components/layout/Sidebar';

interface AiInsightsPageProps {
  onNavigate: (route: NavRoute, plantId?: string) => void;
}

export const AiInsightsPage: React.FC<AiInsightsPageProps> = ({ onNavigate }) => {
  const [plants, setPlants] = useState<Plant[]>([]);
  const [isModelConnected, setIsModelConnected] = useState<boolean>(apiClient.isModelConnected());
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [isModelConnected]);

  const loadData = async () => {
    setLoading(true);
    try {
      const p = await apiClient.getPlants();
      setPlants(p);
      if (isModelConnected) {
        const m = await apiClient.getModelMetrics();
        setMetrics(m);
      } else {
        setMetrics(null);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleToggleModel = (connected: boolean) => {
    apiClient.setModelConnected(connected);
    setIsModelConnected(connected);
  };

  // Prediction distribution
  const predictions = plants.map((p) => p.conditionPrediction).filter(Boolean);
  const conditionCounts: Record<string, number> = {};
  predictions.forEach((pred) => {
    if (pred) {
      conditionCounts[pred.condition] = (conditionCounts[pred.condition] || 0) + 1;
    }
  });

  return (
    <div className="space-y-6">
      {/* Header & Connection Toggle */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
                NVIDIA Jetson Edge Inference
              </span>
              <StatusBadge status={isModelConnected ? 'online' : 'offline'} size="sm" />
            </div>
            <h1 className="text-lg font-bold text-[#1C2B23]">
              Plant Stress & Microclimate ML Diagnostics
            </h1>
          </div>
        </div>

        {/* Toggle to simulate model connected / disconnected states as specified */}
        <div className="flex items-center gap-3 text-xs">
          <span className="text-[#6B786E]">Simulate Hardware Service:</span>
          <button
            onClick={() => handleToggleModel(!isModelConnected)}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all border ${
              isModelConnected
                ? 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8] hover:bg-[#DDE8D8]'
                : 'bg-red-50 text-red-700 border-red-200 hover:bg-red-100'
            }`}
          >
            {isModelConnected ? 'Model Service: Online' : 'Model Service: Offline'}
          </button>
        </div>
      </div>

      {!isModelConnected ? (
        /* Polished "Model Not Connected" Empty State as required by Page 5 */
        <EmptyState
          icon={<Cpu className="w-8 h-8 text-stone-400" />}
          title="Edge ML Model Not Connected"
          description="The Jetson Orin Nano inference service is currently not running or unreachable at the configured model endpoint. Sensor data collection continues unimpeded. Connect or start the local classifier daemon to view stress predictions and explainability maps."
          actionText="Connect Jetson ML Service"
          onAction={() => handleToggleModel(true)}
        />
      ) : (
        <>
          {/* Edge Model Overview Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Classifier Model Version"
              value="v1.2.0-jetson"
              subtitle="Edge XGBoost Pipeline"
              icon={<Cpu className="w-4 h-4 text-[#2E8B57]" />}
              badge={<StatusBadge status="optimal" size="sm" />}
            />
            <MetricCard
              title="Inference Latency"
              value="1.8 ms"
              subtitle="Jetson Orin GPU TensorRT"
              icon={<Sparkles className="w-4 h-4 text-[#2E8B57]" />}
              trend={{ value: 'Sub-millisecond', direction: 'up' }}
            />
            <MetricCard
              title="Test Benchmark Accuracy"
              value="94.2%"
              subtitle="N = 1,420 holdout evaluations"
              icon={<Target className="w-4 h-4 text-[#2E8B57]" />}
              onClick={() => onNavigate('model-performance')}
            />
            <MetricCard
              title="Active Diagnoses"
              value={`${predictions.length} / 16 Plants`}
              subtitle="3 Attention Flags Identified"
              icon={<BrainCircuit className="w-4 h-4 text-[#2E8B57]" />}
              badge={<StatusBadge status="healthy" size="sm" />}
            />
          </div>

          {/* Condition Distribution Overview */}
          <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-bold text-[#1C2B23]">
                  Specimen Condition Distribution
                </h3>
                <p className="text-xs text-[#6B786E]">
                  Inferred phenotype state across 16 monitored indoor flora.
                </p>
              </div>
              <span className="text-xs font-mono text-stone-500">
                100% Ingest Coverage
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 my-2">
              <div className="p-3.5 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8]">
                <div className="flex items-center justify-between text-xs mb-1 text-[#2E8B57] font-semibold">
                  <span>Healthy Control (Cluster 1)</span>
                  <span>{conditionCounts['Healthy'] || 4} Plants</span>
                </div>
                <div className="w-full bg-[#DDE8D8] h-2 rounded-full overflow-hidden">
                  <div className="bg-[#2E8B57] h-full" style={{ width: '25%' }} />
                </div>
                <span className="text-[10px] text-stone-500 mt-1 block">Baseline optimal growth</span>
              </div>

              <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200">
                <div className="flex items-center justify-between text-xs mb-1 text-amber-800 font-semibold">
                  <span>Water Stress (Cluster 2)</span>
                  <span>{conditionCounts['Water Stress'] || 4} Plants</span>
                </div>
                <div className="w-full bg-amber-200 h-2 rounded-full overflow-hidden">
                  <div className="bg-amber-600 h-full" style={{ width: '25%' }} />
                </div>
                <span className="text-[10px] text-amber-700 mt-1 block">Soil moisture &lt; 20% deficit</span>
              </div>

              <div className="p-3.5 rounded-xl bg-yellow-50 border border-yellow-200">
                <div className="flex items-center justify-between text-xs mb-1 text-yellow-800 font-semibold">
                  <span>Light Stress (Cluster 3)</span>
                  <span>{conditionCounts['Light Stress'] || 4} Plants</span>
                </div>
                <div className="w-full bg-yellow-200 h-2 rounded-full overflow-hidden">
                  <div className="bg-yellow-600 h-full" style={{ width: '25%' }} />
                </div>
                <span className="text-[10px] text-yellow-700 mt-1 block">Low PAR &lt; 500 lux shade</span>
              </div>

              <div className="p-3.5 rounded-xl bg-purple-50 border border-purple-200">
                <div className="flex items-center justify-between text-xs mb-1 text-purple-800 font-semibold">
                  <span>Nutrient Stress (Cluster 4)</span>
                  <span>{conditionCounts['Nutrient Stress'] || 4} Plants</span>
                </div>
                <div className="w-full bg-purple-200 h-2 rounded-full overflow-hidden">
                  <div className="bg-purple-600 h-full" style={{ width: '25%' }} />
                </div>
                <span className="text-[10px] text-purple-700 mt-1 block">Nitrogen/Macronutrient depletion</span>
              </div>
            </div>
          </div>

          {/* Intelligence Navigation Cards to SHAP & Performance */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div
              onClick={() => onNavigate('shap-analysis')}
              className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
                <PieChart className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1">
                SHAP Explainability Workspace
              </h3>
              <p className="text-xs text-[#6B786E] leading-relaxed mb-4">
                Explore local waterfall attributions and global beeswarm feature impact distributions without black-box opacity.
              </p>
              <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                Open SHAP Workspace <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </div>

            <div
              onClick={() => onNavigate('model-performance')}
              className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
                <Target className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1">
                Model Performance & Matrix
              </h3>
              <p className="text-xs text-[#6B786E] leading-relaxed mb-4">
                Review holdout test confusion matrix, precision/recall, and per-class evaluation metrics.
              </p>
              <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                View Performance Metrics <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </div>

            <div
              onClick={() => onNavigate('recommendations')}
              className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1">
                Agronomic Recommendations
              </h3>
              <p className="text-xs text-[#6B786E] leading-relaxed mb-4">
                Rule-reviewed, actionable care directives based on empirical thresholds and species profiles.
              </p>
              <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                Review Care Actions <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
