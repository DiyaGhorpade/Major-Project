import React, { useState } from 'react';
import {
  ArrowLeft,
  Sparkles,
  PieChart,
} from 'lucide-react';
import { SensorRecord, UserRecommendationStatus } from '../types';
import { usePlant } from '../hooks/usePlants';
import { useReadings } from '../hooks/useReadings';
import { useRecommendations } from '../hooks/useRecommendations';
import { ErrorState } from '../components/shared/ErrorState';
import { AnomalyTip, EnvStateTip } from '../components/shared/Disclaimers';
import { StatusBadge } from '../components/shared/StatusBadge';
import { SensorReadingCard } from '../components/shared/SensorReadingCard';
import { SensorTrendChart } from '../components/charts/SensorTrendChart';
import { RecommendationCard } from '../components/shared/RecommendationCard';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { formatDateTime, formatRelativeTime } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface PlantDetailPageProps {
  plantId: number | string;
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const PlantDetailPage: React.FC<PlantDetailPageProps> = ({ plantId, onNavigate }) => {
  const numId = typeof plantId === 'string' ? parseInt(plantId.replace('plant-', ''), 10) : plantId;
  const plantQuery = usePlant(numId);
  const readingsQuery = useReadings({ plantId: numId, limit: 30 });
  const recsQuery = useRecommendations({ plantId: numId });
  const plant = plantQuery.data ?? null;
  const plantRecords = readingsQuery.data ?? [];
  const recommendations = recsQuery.data ?? [];
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);

  const handleUpdateRecStatus = (id: string, status: UserRecommendationStatus) => {
    void recsQuery.setStatus(id, status);
  };

  if (plantQuery.loading) {
    return <div className="p-8 text-center text-xs text-stone-500">Loading plant...</div>;
  }

  if (plantQuery.error) {
    return <ErrorState message={plantQuery.error.message} onRetry={() => void plantQuery.refetch()} />;
  }

  if (!plant) {
    return (
      <div className="p-8 text-center bg-white rounded-card border border-[#DCE4D9]">
        <h3 className="text-base font-bold text-[#1C2B23]">Plant Not Found</h3>
        <p className="text-xs text-stone-500 my-2">No pot metadata found for identifier #{plantId}.</p>
        <button
          onClick={() => onNavigate('plants')}
          className="px-4 py-2 bg-[#2E8B57] text-white rounded-lg text-xs font-semibold"
        >
          Return to All 16 Pots
        </button>
      </div>
    );
  }

  const latest = plant.latestReading || plantRecords[0];
  const pred = plant.conditionPrediction;
  const isStressed = plant.stressType !== 'Healthy';

  return (
    <div className="space-y-6">
      {/* Navigation & Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <button
          onClick={() => onNavigate('plants')}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#426B4F] hover:text-[#173F32] transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to All 16 Tulsi Pots</span>
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate('shap-analysis', plant.id)}
            className="px-3.5 py-1.5 bg-[#EDF4EB] hover:bg-[#DDE8D8] text-[#173F32] rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors border border-[#DDE8D8]"
          >
            <PieChart className="w-3.5 h-3.5 text-[#2E8B57]" />
            <span>Explain with AI (SHAP)</span>
          </button>
        </div>
      </div>

      {/* Specimen Profile Dossier Banner */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-6 shadow-card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#DCE4D9] pb-5 mb-5">
          <div className="flex items-start gap-3.5">
            <span className="w-12 h-12 rounded-2xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-extrabold text-xl shrink-0 shadow-sm border border-[#DDE8D8]">
              #{plant.id}
            </span>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
                  {plant.clusterLabel || plant.cluster}
                </span>
                <StatusBadge status={plant.stressType} size="sm" />
              </div>
              <h1 className="text-2xl font-bold text-[#1C2B23] tracking-tight">
                {plant.name}
              </h1>
              <p className="text-xs text-stone-500 italic mt-0.5">
                {plant.variety} • {plant.species} • {plant.location}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 bg-stone-50 px-4 py-3 rounded-xl border border-stone-200 text-xs">
            <div>
              <span className="text-stone-400 block text-[10px] uppercase font-semibold">ESP32 Node</span>
              <span className="font-mono font-bold text-[#1C2B23]">Node {plant.nodeId}</span>
            </div>
            <div className="border-l border-stone-300 pl-4">
              <span className="text-stone-400 block text-[10px] uppercase font-semibold">Sensor Point</span>
              <span className="font-mono font-bold text-[#1C2B23]">Point {plant.samplingPoint ?? 'n/a'}</span>
            </div>
            <div className="border-l border-stone-300 pl-4">
              <span className="text-stone-400 block text-[10px] uppercase font-semibold">Last Checked</span>
              <span className="font-semibold text-[#2E8B57]">
                {latest ? formatRelativeTime(latest.timestamp) : 'Ready'}
              </span>
            </div>
          </div>
        </div>

        <p className="text-xs text-[#6B786E] leading-relaxed max-w-3xl mb-4">
          {plant.description}
        </p>

        {/* Layman Friendly Care Tip Banner */}
        <div className="p-3.5 rounded-xl bg-[#EDF4EB]/80 border border-[#DDE8D8] text-xs text-[#102F26] flex items-start gap-2.5">
          <Sparkles className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
          <div>
            <strong className="font-bold">Simple Care Tip: </strong>
            <span>{plant.simpleCareTip}</span>
          </div>
        </div>
      </div>

      {/* Calibrated Physical Measurements */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">
              Real-Time Pot Sensor Measurements
            </h2>
            <p className="text-xs text-[#6B786E]">
              Calibrated readings from sensors installed inside Pot #{plant.id}.
            </p>
          </div>
          <span className="text-[11px] font-mono text-stone-500">
            Recorded: {latest ? formatDateTime(latest.timestamp) : 'N/A'}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <SensorReadingCard
            type="soil"
            value={latest?.soil}
            optimalRange={plant.optimalRanges.soil}
            timestamp={latest ? formatRelativeTime(latest.timestamp) : undefined}
          />
          <SensorReadingCard
            type="temperature"
            value={latest?.temperature}
            optimalRange={plant.optimalRanges.temperature}
            timestamp={latest ? formatRelativeTime(latest.timestamp) : undefined}
          />
          <SensorReadingCard
            type="humidity"
            value={latest?.humidity}
            optimalRange={plant.optimalRanges.humidity}
            timestamp={latest ? formatRelativeTime(latest.timestamp) : undefined}
          />
          <SensorReadingCard
            type="light"
            value={latest?.light}
            optimalRange={plant.optimalRanges.light}
            timestamp={latest ? formatRelativeTime(latest.timestamp) : undefined}
          />
        </div>
      </div>

      {/* Edge ML Condition Inference Panel */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-[#EDF4EB] text-[#2E8B57]">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#426B4F]">
                Edge Machine Learning Model
              </span>
              <h3 className="text-base font-bold text-[#1C2B23]">
                AI Condition Diagnosis: {pred?.condition || 'Healthy'}
              </h3>
            </div>
          </div>

          <button
            onClick={() => onNavigate('shap-analysis', plant.id)}
            className="px-3.5 py-1.5 rounded-lg bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm"
          >
            <PieChart className="w-3.5 h-3.5" />
            <span>Open SHAP Explanation (Why?)</span>
          </button>
        </div>

        {pred && (
          <div className="p-4 rounded-xl bg-[#EDF4EB]/50 border border-[#DDE8D8] space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <span className="text-[11px] text-[#6B786E] block mb-1">Heuristic stress-type guess:</span>
                <span className="text-lg font-bold text-[#102F26]">{pred.condition ?? 'Not determined'}</span>
              </div>
              <div>
                <span className="text-[11px] text-[#6B786E] block mb-1">AI Certainty:</span>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-mono font-bold text-[#2E8B57]">
                    {pred.confidence === null ? 'n/a' : `${(pred.confidence * 100).toFixed(0)}%`}
                  </span>
                  <div className="w-24 bg-stone-200 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-[#2E8B57] h-full rounded-full"
                      style={{ width: `${(pred.confidence ?? 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              <div>
                <span className="text-[11px] text-[#6B786E] block mb-1">Inference Engine:</span>
                <span className="font-mono text-xs text-stone-700">
                  Model {pred.modelVersion ?? 'n/a'}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-[11px] text-[#6B786E]">
              {pred.envState && (
                <span>
                  {pred.envState.label} <EnvStateTip />
                </span>
              )}
              {pred.anomaly.isAnomaly !== null && (
                <span>
                  Reading check: {pred.anomaly.isAnomaly ? 'unusual reading' : 'normal'} <AnomalyTip />
                </span>
              )}
            </div>

            {/* Simple Layman Explanation of the AI Decision */}
            {pred.simpleExplanation && (
              <div className="pt-3 border-t border-[#DDE8D8] text-xs text-[#1C2B23] leading-relaxed">
                <strong className="text-[#2E8B57]">In Plain Terms: </strong>
                {pred.simpleExplanation}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Historical Trend Chart */}
      <div className="space-y-3">
        <h2 className="text-base font-bold text-[#1C2B23]">
          Telemetry Trends Over Time for Pot #{plant.id}
        </h2>
        <SensorTrendChart
          records={plantRecords}
          title={`Readings History — Pot #${plant.id} (${plant.name})`}
          optimalRange={plant.optimalRanges}
          height={300}
        />
      </div>

      {/* Care Directives */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">
              Recommended Care Actions for Pot #{plant.id}
            </h2>
            <p className="text-xs text-[#6B786E]">
              Simple, evidence-based recommendations based on Pot #{plant.id}'s actual sensor readings.
            </p>
          </div>
        </div>

        {recommendations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recommendations.map((rec) => (
              <RecommendationCard
                key={rec.id}
                recommendation={rec}
                onUpdateStatus={handleUpdateRecStatus}
              />
            ))}
          </div>
        ) : (
          <div className="p-6 rounded-card border border-[#DCE4D9] bg-white text-center text-xs text-stone-500">
            Pot #{plant.id} is healthy! All sensor values are in the optimal green zone.
          </div>
        )}
      </div>

      {/* Record Inspector Drawer */}
      <RecordDetailDrawer
        record={selectedRecord}
        onClose={() => setSelectedRecord(null)}
      />
    </div>
  );
};
