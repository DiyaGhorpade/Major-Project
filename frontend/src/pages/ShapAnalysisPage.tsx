import React, { useState } from 'react';
import {
  PieChart,
  BarChart3,
  Sliders,
  AlertCircle,
} from 'lucide-react';
import { usePlants } from '../hooks/usePlants';
import { useGlobalShap, useLocalShap } from '../hooks/useData';
import { ShapWaterfallChart } from '../components/charts/ShapWaterfallChart';
import { ShapBeeswarmChart } from '../components/charts/ShapBeeswarmChart';
import { EmptyState, RunEmptyState } from '../components/shared/EmptyState';

interface ShapAnalysisPageProps {
  initialPlantId?: number | string;
  onNavigate: (route: any, plantId?: number | string) => void;
}

export const ShapAnalysisPage: React.FC<ShapAnalysisPageProps> = ({
  initialPlantId = 5,
  onNavigate,
}) => {
  const [activeTab, setActiveTab] = useState<'local' | 'global'>('local');
  const [selectedPlantId, setSelectedPlantId] = useState<string>(String(initialPlantId));
  const plants = usePlants().data ?? [];
  const localQuery = useLocalShap(selectedPlantId);
  const globalQuery = useGlobalShap();
  const localShap = localQuery.data;
  const globalShap = globalQuery.data?.data ?? null;
  const globalHint = globalQuery.data?.unavailable?.hint ?? null;

  const selectedPlant = plants.find((p) => p.id === Number(selectedPlantId)) || plants[4] || plants[0];

  return (
    <div className="space-y-6">
      {/* Workspace Navigation Header & Tabs */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              Explainable AI (XAI)
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              TreeSHAP / Jetson Edge
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            SHAP (SHapley Additive exPlanations) Diagnostic Workspace
          </h1>
        </div>

        {/* Local vs Global Tabs */}
        <div className="inline-flex rounded-lg bg-[#EDF4EB] p-1 text-xs font-medium text-[#173F32]">
          <button
            onClick={() => setActiveTab('local')}
            className={`px-3.5 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
              activeTab === 'local'
                ? 'bg-white text-[#102F26] shadow-sm font-bold'
                : 'hover:text-[#102F26]'
            }`}
          >
            <PieChart className="w-3.5 h-3.5" />
            <span>Local Explanation (Single Specimen)</span>
          </button>
          <button
            onClick={() => setActiveTab('global')}
            className={`px-3.5 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
              activeTab === 'global'
                ? 'bg-white text-[#102F26] shadow-sm font-bold'
                : 'hover:text-[#102F26]'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Global Feature Impact (Cohort)</span>
          </button>
        </div>
      </div>

      {/* Mandatory Top Disclaimer Banner */}
      <div className="p-3.5 rounded-xl bg-[#F6F7F0] border border-[#DCE4D9] text-xs text-[#1C2B23] flex items-center gap-2.5">
        <AlertCircle className="w-4 h-4 text-[#2E8B57] shrink-0" />
        <span className="leading-normal">
          <strong>Interpretability Standard:</strong> SHAP explains how features influence the model's prediction. It does not establish causality or independently identify the biological root cause.
        </span>
      </div>

      {activeTab === 'local' ? (
        /* Local SHAP View */
        <div className="space-y-6">
          {/* Plant & Target Selection Controller */}
          <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs font-bold text-[#173F32] flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#2E8B57]" /> Focus Specimen:
              </span>
              <select
                value={selectedPlantId}
                onChange={(e) => setSelectedPlantId(e.target.value)}
                className="text-xs font-bold bg-[#EDF4EB] border border-[#DDE8D8] rounded-lg px-3 py-1.5 text-[#102F26] outline-none cursor-pointer"
              >
                {plants.map((p) => (
                  <option key={p.id} value={p.id}>
                    Pot #{p.id}: {p.name} ({p.cluster})
                  </option>
                ))}
              </select>

              {selectedPlant && (
                <span className="text-xs text-[#6B786E]">
                  Species: <span className="italic">{selectedPlant.species}</span> (Node {selectedPlant.nodeId})
                </span>
              )}
            </div>

            <div className="text-xs font-mono text-stone-500">
              Evaluated on latest validated reading • Output Space: Class Probability [0, 1]
            </div>
          </div>

          {/* Local Waterfall Chart */}
          {localShap ? (
            <ShapWaterfallChart explanation={localShap} />
          ) : (
            <RunEmptyState title="No explanation yet" />
          )}
        </div>
      ) : (
        /* Global SHAP View */
        <div className="space-y-6">
          {globalShap ? (
            <ShapBeeswarmChart data={globalShap} />
          ) : (
            <EmptyState
              title="Global SHAP Data Unavailable"
              description={
                globalHint
                  ? `The global SHAP report has not been generated yet. Run: ${globalHint}`
                  : 'Global SHAP report not available.'
              }
            />
          )}
        </div>
      )}
    </div>
  );
};
