import React, { useState } from 'react';
import {
  PieChart,
  BarChart3,
  Sliders,
} from 'lucide-react';
import { usePlants } from '../hooks/usePlants';
import { useGlobalShap, useLocalShap } from '../hooks/useData';
import { ShapWaterfallChart } from '../components/charts/ShapWaterfallChart';
import { ShapBeeswarmChart } from '../components/charts/ShapBeeswarmChart';
import { EmptyState, RunEmptyState } from '../components/shared/EmptyState';
import { ErrorState } from '../components/shared/ErrorState';
import { NavRoute } from '../components/layout/Sidebar';
import { formatDateTime } from '../lib/utils';

interface ShapAnalysisPageProps {
  initialPlantId?: number | string;
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const ShapAnalysisPage: React.FC<ShapAnalysisPageProps> = ({ initialPlantId }) => {
  const [activeTab, setActiveTab] = useState<'local' | 'global'>('local');
  const [pickedPlantId, setSelectedPlantId] = useState<string | null>(
    initialPlantId === undefined ? null : String(initialPlantId).replace('plant-', ''),
  );
  const plants = usePlants().data ?? [];
  // No explicit pick (and none passed in): the first plant from the data, not a hard-coded pot.
  const selectedPlantId = pickedPlantId ?? (plants[0] ? String(plants[0].id) : '');
  const localQuery = useLocalShap(selectedPlantId === '' ? null : selectedPlantId);
  const globalQuery = useGlobalShap();
  const localShap = localQuery.data;
  const globalShap = globalQuery.data?.data ?? null;
  const globalHint = globalQuery.data?.unavailable?.hint ?? null;

  const selectedPlant = plants.find((p) => p.id === Number(selectedPlantId));

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
              Model explanation (SHAP, log-odds)
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            SHAP (SHapley Additive exPlanations) Explainability Workspace
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
                {plants.length === 0 && <option value={selectedPlantId}>{selectedPlantId ? `Pot #${selectedPlantId}` : 'Loading plants...'}</option>}
                {plants.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.cluster})
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
              {localShap
                ? `Explains reading ${localShap.record_id} (${formatDateTime(localShap.timestamp)} IST) • Units: log-odds`
                : 'Explains the plant\'s newest stored prediction • Units: log-odds'}
            </div>
          </div>

          {/* Local Waterfall Chart */}
          {localQuery.error && !localShap ? (
            <ErrorState message={localQuery.error.message} onRetry={() => void localQuery.refetch()} />
          ) : localQuery.loading ? (
            <EmptyState title="Loading explanation..." description="" />
          ) : localShap ? (
            <ShapWaterfallChart explanation={localShap} />
          ) : (
            <RunEmptyState title="No explanation yet" />
          )}
        </div>
      ) : (
        /* Global SHAP View */
        <div className="space-y-6">
          {globalQuery.error && !globalShap ? (
            <ErrorState message={globalQuery.error.message} onRetry={() => void globalQuery.refetch()} />
          ) : globalQuery.loading ? (
            <EmptyState title="Loading global SHAP report..." description="" />
          ) : globalShap ? (
            <ShapBeeswarmChart data={globalShap} />
          ) : (
            <EmptyState
              title="Global SHAP Data Unavailable"
              description={
                <span data-testid="global-shap-unavailable">
                  The global SHAP report has not been generated yet.
                  {globalHint ? (
                    <>
                      {' '}Run this command once:
                      <code className="block mt-3 p-2 rounded bg-stone-100 text-[11px] text-left text-stone-800 break-all font-mono">{globalHint}</code>
                    </>
                  ) : null}
                </span>
              }
            />
          )}
        </div>
      )}
    </div>
  );
};
