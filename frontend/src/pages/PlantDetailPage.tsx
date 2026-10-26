import React, { useState } from 'react';
import { ArrowLeft, Sparkles, PieChart } from 'lucide-react';
import { SensorRecord } from '../types';
import { usePlantMonitoring } from '../hooks/usePlants';
import { ErrorState } from '../components/shared/ErrorState';
import { RunEmptyState } from '../components/shared/EmptyState';
import { DISCLAIMERS } from '../components/shared/disclaimerText';
import { SensorReadingCard } from '../components/shared/SensorReadingCard';
import { SensorTrendChart } from '../components/charts/SensorTrendChart';
import { RecommendationCard } from '../components/shared/RecommendationCard';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { AnomalyFlag, EnvStateLabel, PlantStatusBadge, SeverityBadge } from '../components/monitoring/PredictionBadges';
import { ProbabilityBars } from '../components/monitoring/ProbabilityBars';
import { DatasetLabelsBlock } from '../components/monitoring/DatasetLabelsBlock';
import { useRecommendationStatus } from '../hooks/useRecommendationStatus';
import { describeRecommendationError } from '../components/ai/recommendationHelpers';
import { TimeRangeButtons } from '../components/monitoring/TimeRangeButtons';
import { filterByRange, formatConfidence, type TimeRange } from '../components/monitoring/format';
import { formatDateTime, orNA } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface PlantDetailPageProps {
  plantId: number | string;
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

const STRIP_COLORS: Record<string, string> = {
  healthy: '#79B86A',
  low: '#E5C84A',
  moderate: '#E8A33C',
  high: '#D9534F',
};
const HISTORY_ROWS = 12;

export const PlantDetailPage: React.FC<PlantDetailPageProps> = ({ plantId, onNavigate }) => {
  const numId = typeof plantId === 'string' ? parseInt(plantId.replace('plant-', ''), 10) : plantId;
  const monitoringQuery = usePlantMonitoring(Number.isFinite(numId) ? numId : null, 336);
  const actions = useRecommendationStatus();
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);
  const [range, setRange] = useState<TimeRange>('7d');
  const [showAllHistory, setShowAllHistory] = useState(false);

  if (monitoringQuery.loading) {
    return <div className="p-8 text-center text-xs text-stone-500">Loading plant...</div>;
  }

  if (monitoringQuery.error && !monitoringQuery.data) {
    return <ErrorState message={monitoringQuery.error.message} onRetry={() => void monitoringQuery.refetch()} />;
  }

  const monitoring = monitoringQuery.data;
  const plant = monitoring?.plant ?? null;

  if (!monitoring || !plant) {
    return (
      <div className="p-8 text-center bg-white rounded-card border border-[#DCE4D9]">
        <h3 className="text-base font-bold text-[#1C2B23]">Plant Not Found</h3>
        <p className="text-xs text-stone-500 my-2">No pot metadata found for identifier #{plantId}.</p>
        <button onClick={() => onNavigate('plants')} className="px-4 py-2 bg-[#2E8B57] text-white rounded-lg text-xs font-semibold">
          Return to All Pots
        </button>
      </div>
    );
  }

  const latest = plant.latestReading;
  const pred = plant.conditionPrediction;
  const seriesInRange = filterByRange(monitoring.series, range);
  const history = filterByRange(monitoring.predictionHistory, range);
  const historyNewestFirst = [...history].reverse();
  const hasPredictions = pred !== null || monitoring.predictionHistory.length > 0;
  const anomalyCount = history.filter((h) => h.isAnomaly === true).length;
  const envCounts = new Map<string, number>();
  for (const h of history) if (h.envState) envCounts.set(h.envState.label, (envCounts.get(h.envState.label) ?? 0) + 1);
  const recommendations = monitoring.recommendations;

  return (
    <div className="space-y-6">
      {monitoringQuery.error && <ErrorState title="Could not refresh (showing earlier data)" message={monitoringQuery.error.message} onRetry={() => void monitoringQuery.refetch()} />}

      {/* Navigation & Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <button
          onClick={() => onNavigate('plants')}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#426B4F] hover:text-[#173F32] transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to All Tulsi Pots</span>
        </button>

        <button
          onClick={() => onNavigate('shap-analysis', plant.id)}
          className="px-3.5 py-1.5 bg-[#EDF4EB] hover:bg-[#DDE8D8] text-[#173F32] rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors border border-[#DDE8D8]"
        >
          <PieChart className="w-3.5 h-3.5 text-[#2E8B57]" />
          <span>Explain with AI (SHAP)</span>
        </button>
      </div>

      {/* Profile banner */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-6 shadow-card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#DCE4D9] pb-5 mb-5">
          <div className="flex items-start gap-3.5">
            <span className="w-12 h-12 rounded-2xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-extrabold text-xl shrink-0 shadow-sm border border-[#DDE8D8]">
              #{plant.id}
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]" title="Experimental group (designed treatment), not a model cluster">
                  {plant.clusterLabel || plant.cluster}
                </span>
                <PlantStatusBadge status={plant.status} />
              </div>
              <h1 className="text-2xl font-bold text-[#1C2B23] tracking-tight">{plant.name}</h1>
              <p className="text-xs text-stone-500 italic mt-0.5">
                {plant.variety} • {plant.species} • {plant.location}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 bg-stone-50 px-4 py-3 rounded-xl border border-stone-200 text-xs">
            <div>
              <span className="text-stone-400 block text-[10px] uppercase font-semibold">Sensor Node</span>
              <span className="font-mono font-bold text-[#1C2B23]">Node {plant.nodeId}</span>
            </div>
            <div className="border-l border-stone-300 pl-4">
              <span className="text-stone-400 block text-[10px] uppercase font-semibold">Sensor Point</span>
              <span className="font-mono font-bold text-[#1C2B23]">Point {orNA(plant.samplingPoint)}</span>
            </div>
            <div className="border-l border-stone-300 pl-4">
              <span className="text-stone-400 block text-[10px] uppercase font-semibold">Last Reading (IST)</span>
              <span className="font-semibold text-[#2E8B57]">{latest ? formatDateTime(latest.timestamp) : 'n/a'}</span>
            </div>
          </div>
        </div>

        <p className="text-xs text-[#6B786E] leading-relaxed max-w-3xl mb-4">{plant.description}</p>

        <div className="p-3.5 rounded-xl bg-[#EDF4EB]/80 border border-[#DDE8D8] text-xs text-[#102F26] flex items-start gap-2.5">
          <Sparkles className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
          <div>
            <strong className="font-bold">Simple Care Tip: </strong>
            <span>{plant.simpleCareTip}</span>
          </div>
        </div>
      </div>

      {/* Latest measurements */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">Latest Pot Sensor Measurements</h2>
            <p className="text-xs text-[#6B786E]">Most recent stored reading for Pot #{plant.id} (Most recent readings).</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-stone-500">Recorded: {latest ? formatDateTime(latest.timestamp) : 'n/a'} IST</span>
            {latest && (
              <button
                onClick={() => setSelectedRecord(latest)}
                className="px-2.5 py-1 text-xs rounded bg-[#EDF4EB] text-[#2E8B57] font-semibold hover:bg-[#DDE8D8] transition-colors"
              >
                Inspect record
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {(['soil', 'temperature', 'humidity', 'light'] as const).map((k) => (
            <SensorReadingCard
              key={k}
              type={k}
              value={latest?.[k]}
              optimalRange={plant.optimalRanges[k]}
              timestamp={latest ? formatDateTime(latest.timestamp) : undefined}
              readingTime={latest?.timestamp}
            />
          ))}
        </div>
        {latest && latest.status !== 'accepted' && (
          <p className="text-[11px] text-amber-800">
            Latest reading is {latest.status}
            {latest.validation_error ? ` (${latest.validation_error})` : ''}.
          </p>
        )}
      </div>

      {/* Model output */}
      {!hasPredictions ? (
        <RunEmptyState title="No model results for this pot yet" className="my-2" />
      ) : (
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4" data-testid="ml-panel">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-[#EDF4EB] text-[#2E8B57]">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#426B4F]">Machine Learning Model Output</span>
                <h3 className="text-base font-bold text-[#1C2B23] flex items-center gap-2">
                  Severity (ML): {pred ? <SeverityBadge severity={pred.severity} /> : 'n/a'}
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
                  <span className="text-[11px] text-[#6B786E] mb-1 flex items-center gap-1">
                    Stress type guess
                  </span>
                  <span className="text-lg font-bold text-[#102F26]">{pred.condition ?? 'Not determined'}</span>
                </div>
                <div>
                  <span className="text-[11px] text-[#6B786E] block mb-1">Severity confidence (ML)</span>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-mono font-bold text-[#2E8B57]">{formatConfidence(pred.confidence)}</span>
                    <div className="w-24 bg-stone-200 h-2 rounded-full overflow-hidden">
                      <div className="bg-[#2E8B57] h-full rounded-full" style={{ width: `${(pred.confidence ?? 0) * 100}%` }} />
                    </div>
                  </div>
                </div>
                <div>
                  <span className="text-[11px] text-[#6B786E] block mb-1">Model</span>
                  <span className="font-mono text-xs text-stone-700">
                    {orNA(pred.modelVersion)} • {formatDateTime(pred.timestamp)}
                  </span>
                </div>
              </div>
              <div className="max-w-md">
                <ProbabilityBars probabilities={pred.probabilities} />
              </div>
              {pred.simpleExplanation && (
                <div className="pt-3 border-t border-[#DDE8D8] text-xs text-[#1C2B23] leading-relaxed">
                  <strong className="text-[#2E8B57]">In Plain Terms: </strong>
                  {pred.simpleExplanation}
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-xl border border-amber-200 bg-amber-50/40 p-4 text-xs space-y-1.5" data-testid="anomaly-panel">
              <h4 className="font-bold text-[#1C2B23] flex items-center gap-1.5">
                Anomaly check
              </h4>
              <div className="text-sm">
                <AnomalyFlag isAnomaly={pred?.anomaly?.isAnomaly} />
              </div>
              <div className="text-stone-600">
                Anomaly score: <span className="font-mono">{pred?.anomaly?.score == null ? 'n/a' : pred.anomaly.score.toFixed(3)}</span>
              </div>
              <div className="text-stone-600">
                {anomalyCount} of {history.length} analyzed readings in view flagged as unusual.
              </div>
            </div>

            <div className="rounded-xl border border-sky-200 bg-sky-50/40 p-4 text-xs space-y-1.5" data-testid="envstate-panel">
              <h4 className="font-bold text-[#1C2B23] flex items-center gap-1.5">
                Environmental state
              </h4>
              <div className="text-sm">
                <EnvStateLabel envState={pred?.envState} />
              </div>
              <div className="text-stone-600">
                Distance to group centre: <span className="font-mono">{pred?.envState?.distance == null ? 'n/a' : pred.envState.distance.toFixed(2)}</span>
              </div>
              <div className="text-stone-600">
                {envCounts.size === 0
                  ? 'No state history in view.'
                  : `In view: ${[...envCounts.entries()].map(([label, n]) => `${label.replace('Environmental state ', '')} (${n})`).join(', ')}`}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Trend + prediction history */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-bold text-[#1C2B23]">Telemetry Trends Over Time for Pot #{plant.id}</h2>
          <TimeRangeButtons value={range} onChange={setRange} />
        </div>
        <SensorTrendChart
          records={seriesInRange}
          title={`Readings History: Pot #${plant.id} (${plant.name})`}
          optimalRange={plant.optimalRanges}
          height={300}
        />
      </div>

      {hasPredictions && (
        <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden" data-testid="prediction-history">
          <div className="p-4 border-b border-[#DCE4D9]">
            <h3 className="text-sm font-bold text-[#1C2B23]">Prediction History (ML)</h3>
            <p className="text-xs text-[#6B786E]">
              Severity per analyzed reading, oldest to newest. Ringed cells are unusual readings.
            </p>
            <div className="mt-3 flex flex-wrap gap-0.5" aria-label="Severity strip">
              {history.map((h) => (
                <span
                  key={h.record_id}
                  title={`${formatDateTime(h.timestamp)}: ${h.severity ?? h.status}${h.isAnomaly ? ' (unusual reading)' : ''}`}
                  className={`h-4 w-2 rounded-sm ${h.isAnomaly ? 'ring-1 ring-black/70' : ''}`}
                  style={{ backgroundColor: h.severity ? STRIP_COLORS[h.severity] : '#D6D3D1' }}
                />
              ))}
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-stone-500">
              {Object.entries(STRIP_COLORS).map(([k, c]) => (
                <span key={k} className="inline-flex items-center gap-1 capitalize">
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: c }} />
                  {k}
                </span>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-xs text-left">
              <thead className="sticky top-0 bg-[#F6F7F0] border-b border-[#DCE4D9]">
                <tr className="text-[#6B786E]">
                  <th className="py-2.5 px-4 font-semibold">Time (IST)</th>
                  <th className="py-2.5 px-4 font-semibold">Severity (ML)</th>
                  <th className="py-2.5 px-4 font-semibold">Probabilities H / L / M / Hi</th>
                  <th className="py-2.5 px-4 font-semibold">Stress type guess</th>
                  <th className="py-2.5 px-4 font-semibold">Reading check</th>
                  <th className="py-2.5 px-4 font-semibold">Env. state</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DCE4D9]/60">
                {historyNewestFirst.length === 0 && (
                  <tr>
                    <td colSpan={6} className="py-5 px-4 text-center text-stone-500">
                      No predictions in this range.
                    </td>
                  </tr>
                )}
                {(showAllHistory ? historyNewestFirst : historyNewestFirst.slice(0, HISTORY_ROWS)).map((h) => (
                  <tr key={h.record_id}>
                    <td className="py-2 px-4 font-mono text-[#1C2B23]">{formatDateTime(h.timestamp)}</td>
                    <td className="py-2 px-4">
                      {h.status === 'ok' ? (
                        <span className="inline-flex items-center gap-1.5">
                          <SeverityBadge severity={h.severity} />
                          <span className="font-mono text-stone-500">{formatConfidence(h.confidence)}</span>
                        </span>
                      ) : (
                        <span className="text-stone-500">{h.status}</span>
                      )}
                    </td>
                    <td className="py-2 px-4 font-mono text-stone-600">
                      {h.probabilities
                        ? ['healthy', 'low', 'moderate', 'high'].map((k) => formatConfidence(h.probabilities?.[k])).join(' / ')
                        : 'n/a'}
                    </td>
                    <td className="py-2 px-4 text-stone-700">{orNA(h.condition)}</td>
                    <td className="py-2 px-4">
                      <AnomalyFlag isAnomaly={h.isAnomaly} />
                    </td>
                    <td className="py-2 px-4 text-stone-700">{h.envState ? h.envState.label.replace('Environmental state ', 'Group ') : 'n/a'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {historyNewestFirst.length > HISTORY_ROWS && (
            <div className="p-3 border-t border-[#DCE4D9] text-center">
              <button onClick={() => setShowAllHistory(!showAllHistory)} className="text-xs font-bold text-[#2E8B57] hover:underline">
                {showAllHistory ? 'Show fewer' : `Show all ${historyNewestFirst.length} predictions`}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Dataset labels (never mixed with model output) */}
      <DatasetLabelsBlock labels={latest?.datasetLabels} />

      {/* Care Directives */}
      <div className="space-y-3">
        <div>
          <h2 className="text-base font-bold text-[#1C2B23]">Recommended Care Actions for Pot #{plant.id}</h2>
        </div>

        {actions.error && (
          <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800 flex items-start justify-between gap-2">
            <span>{describeRecommendationError(actions.error, recommendations)}</span>
            <button onClick={actions.clearError} className="font-semibold underline">
              Dismiss
            </button>
          </div>
        )}

        {recommendations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recommendations.map((rec) => (
              <RecommendationCard
                key={rec.id}
                recommendation={rec}
                pending={actions.isPending(rec.id)}
                onUpdateStatus={(id, status) => void actions.update(id, status)}
              />
            ))}
          </div>
        ) : hasPredictions ? (
          <div className="p-6 rounded-card border border-[#DCE4D9] bg-white text-center text-xs text-stone-500">
            No active recommendations for Pot #{plant.id}.
          </div>
        ) : (
          <div className="p-6 rounded-card border border-[#DCE4D9] bg-white text-center text-xs text-stone-500">
            Recommendations appear after the analysis has run. {DISCLAIMERS.runPrompt}
          </div>
        )}
      </div>

      <RecordDetailDrawer record={selectedRecord} onClose={() => setSelectedRecord(null)} />
    </div>
  );
};
