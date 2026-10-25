import React, { useState } from 'react';
import {
  Cpu,
  Database,
  CheckCircle2,
  History,
  Activity,
  ArrowRight,
  ShieldCheck,
  Droplets,
  Sun,
  Sprout,
  HeartPulse,
  Sparkles,
} from 'lucide-react';
import { SensorRecord, ClusterSummary } from '../types';
import { usePlants, usePlantMonitoring } from '../hooks/usePlants';
import { useReadings } from '../hooks/useReadings';
import { useOverview } from '../hooks/useData';
import { ErrorState } from '../components/shared/ErrorState';
import { RunEmptyState } from '../components/shared/EmptyState';
import { MetricCard } from '../components/shared/MetricCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { RecommendationCard } from '../components/shared/RecommendationCard';
import { AnomalyTip } from '../components/shared/Disclaimers';
import { DISCLAIMERS } from '../components/shared/disclaimerText';
import { SensorTrendChart } from '../components/charts/SensorTrendChart';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { SeverityBadge } from '../components/monitoring/PredictionBadges';
import { useRecommendationStatus } from '../hooks/useRecommendationStatus';
import { describeRecommendationError } from '../components/ai/recommendationHelpers';
import { TimeRangeButtons } from '../components/monitoring/TimeRangeButtons';
import { filterByRange, formatPlantIds, type TimeRange } from '../components/monitoring/format';
import { formatDateTime, formatCount, formatPercent, formatTemp, formatLux, orNA } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface OverviewPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

// Visual treatment per experimental group (by position). The group names/labels themselves come from the API.
const CLUSTER_STYLES = [
  { Icon: CheckCircle2, card: 'border-[#DCE4D9] hover:border-[#2E8B57]', icon: 'bg-[#EDF4EB] text-[#2E8B57]', title: 'text-[#1C2B23]', pill: 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]' },
  { Icon: Droplets, card: 'border-amber-200 hover:border-amber-400', icon: 'bg-amber-50 text-amber-700', title: 'text-amber-950', pill: 'bg-amber-100 text-amber-900 border-amber-300' },
  { Icon: Sun, card: 'border-yellow-200 hover:border-yellow-400', icon: 'bg-yellow-50 text-yellow-700', title: 'text-yellow-950', pill: 'bg-yellow-100 text-yellow-900 border-yellow-300' },
  { Icon: Sprout, card: 'border-purple-200 hover:border-purple-400', icon: 'bg-purple-50 text-purple-700', title: 'text-purple-950', pill: 'bg-purple-100 text-purple-900 border-purple-300' },
];

function clusterBadge(c: ClusterSummary, hasPredictions: boolean): string {
  if (!hasPredictions) return 'Not analyzed yet';
  const healthy = c.severityCounts.healthy ?? 0;
  return `${healthy}/${c.plants} rated healthy`;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate }) => {
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');
  const [trendPlant, setTrendPlant] = useState<number | null>(null);

  const overviewQuery = useOverview();
  const plantsQuery = usePlants();
  const recentQuery = useReadings({ limit: 8 });
  const actions = useRecommendationStatus();

  const overview = overviewQuery.data;
  const plants = plantsQuery.data ?? [];
  const recentRecords = recentQuery.data ?? [];
  const trendPlantId = trendPlant ?? plants[0]?.id ?? null;
  const trendQuery = usePlantMonitoring(trendPlantId, 336);
  const trend = trendQuery.data;

  const loadError = overviewQuery.error ?? plantsQuery.error ?? recentQuery.error;

  const totals = overview?.totals;
  const hasPredictions = (totals?.predictions.ok ?? 0) > 0;
  const readingsTotal = totals?.readings ?? recentQuery.total;
  const flaggedCount = totals?.flaggedReadings ?? 0;
  const acceptedPct = readingsTotal > 0 ? (((readingsTotal - flaggedCount) / readingsTotal) * 100).toFixed(1) : null;

  const devices = overview?.devicesSummary;
  const deviceStatus = !devices ? 'unknown' : devices.online > 0 ? 'online' : devices.offline > 0 ? 'offline' : 'unknown';
  const session = overview?.latestSession ?? null;
  const statusCounts = overview?.plantStatusCounts ?? {};

  const trendSeries = trend ? filterByRange(trend.series, timeRange) : [];

  return (
    <div className="space-y-6">
      {loadError && (
        <ErrorState
          message={loadError.message}
          onRetry={() => {
            void overviewQuery.refetch();
            void plantsQuery.refetch();
            void recentQuery.refetch();
          }}
        />
      )}

      {/* Top Banner */}
      <div className="bg-gradient-to-r from-[#102F26] to-[#173F32] rounded-card p-6 text-white shadow-lg flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#79B86A]" />
            <span className="text-xs font-bold uppercase tracking-wider text-[#DDE8D8]">Bundled dataset</span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold tracking-tight">Tulsi Phenotype Stress Telemetry Dashboard</h1>
          <p className="text-xs text-[#DDE8D8]/80 mt-1 max-w-2xl leading-relaxed">
            {totals
              ? `${totals.plants} monitored Tulsi pots across ${overview?.clusterSummaries.length ?? 0} experimental groups, ${formatCount(totals.readings)} stored readings.`
              : 'Monitored Tulsi pots across the dataset’s experimental groups.'}{' '}
            Groups are the dataset&apos;s designed treatments, not model clusters.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => onNavigate('live-monitoring')}
            className="px-3.5 py-2 rounded-lg bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Latest Readings</span>
          </button>
          <button
            onClick={() => onNavigate('recommendations')}
            className="px-3.5 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold flex items-center gap-1.5 transition-all"
          >
            <HeartPulse className="w-3.5 h-3.5 text-[#79B86A]" />
            <span>Plant Care Actions</span>
          </button>
          <button
            onClick={() => onNavigate('data-logs')}
            className="px-3.5 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-semibold flex items-center gap-1.5 transition-all"
          >
            <Database className="w-3.5 h-3.5" />
            <span>Data Logs</span>
          </button>
        </div>
      </div>

      {overviewQuery.loading && !overview && <div className="p-6 text-center text-xs text-stone-500">Loading overview...</div>}

      {/* Experimental group cards (real per-group data) */}
      {overview && (
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-[#6B786E] mb-3 flex items-center gap-1.5">
            At-A-Glance: {totals?.plants ?? 0} Tulsi Pots Across {overview.clusterSummaries.length} Experimental Groups
            {overviewQuery.refreshing && <span className="normal-case font-normal text-stone-400">(refreshing...)</span>}
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {overview.clusterSummaries.map((c, i) => {
              const style = CLUSTER_STYLES[i % CLUSTER_STYLES.length];
              const Icon = style.Icon;
              const { soil, temperature, humidity } = c.averages;
              return (
                <div
                  key={c.cluster}
                  data-testid={`cluster-card-${c.cluster}`}
                  onClick={() => onNavigate('plants')}
                  className={`p-4 rounded-xl bg-white border shadow-card cursor-pointer transition-all flex flex-col gap-2 ${style.card}`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-3 min-w-0 flex-1 basis-40">
                      <div className={`w-10 h-10 shrink-0 rounded-xl flex items-center justify-center font-bold ${style.icon}`}>
                        <Icon className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <span className={`text-xs font-bold block break-words ${style.title}`}>{c.clusterLabel}</span>
                        <span className="text-[11px] text-[#6B786E]">{formatPlantIds(c.plantIds)}</span>
                      </div>
                    </div>
                    <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full border whitespace-nowrap shrink-0 ${style.pill}`}>
                      {clusterBadge(c, hasPredictions)}
                    </span>
                  </div>
                  <div className="text-[11px] text-[#6B786E] leading-snug">
                    Avg soil {formatPercent(soil)} • temp {formatTemp(temperature)} • humidity {formatPercent(humidity)}
                  </div>
                  {hasPredictions && (
                    <div className="text-[11px] text-stone-600">
                      Most common stress-type guess (heuristic): <strong>{orNA(c.topCondition)}</strong>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Operational Metrics Row */}
      {overview && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Devices"
            value={devices ? `${devices.total} devices` : 'n/a'}
            subtitle={
              devices && devices.online === 0
                ? 'No live heartbeat (historical dataset)'
                : devices
                  ? `${devices.online} online • ${devices.offline} offline • ${devices.unknown} unknown`
                  : undefined
            }
            icon={<Cpu className="w-4 h-4 text-[#2E8B57]" />}
            badge={<StatusBadge status={deviceStatus} size="sm" />}
            onClick={() => onNavigate('devices-network')}
          />

          <MetricCard
            title="Plant Status (Model)"
            value={hasPredictions ? `${statusCounts.healthy ?? 0} rated healthy` : 'Not analyzed yet'}
            subtitle={
              hasPredictions
                ? `${statusCounts.attention ?? 0} attention • ${statusCounts.critical ?? 0} critical`
                : DISCLAIMERS.runPrompt
            }
            icon={<Sparkles className="w-4 h-4 text-[#2E8B57]" />}
            onClick={() => onNavigate('plants')}
          />

          <MetricCard
            title="Latest Collection Pass"
            value={session ? `Session #${session.session_id}` : 'No sessions'}
            subtitle={session ? `${formatCount(session.received_records)} of ${formatCount(session.expected_records)} readings received` : 'n/a'}
            icon={<History className="w-4 h-4 text-[#2E8B57]" />}
            badge={session ? <StatusBadge status={session.status} size="sm" /> : undefined}
            onClick={() => onNavigate('sessions')}
          />

          <MetricCard
            title="Ingestion Integrity"
            value={acceptedPct === null ? 'n/a' : `${acceptedPct}% Accepted`}
            subtitle={`${formatCount(readingsTotal)} records logged (${formatCount(flaggedCount)} flagged)`}
            icon={<ShieldCheck className="w-4 h-4 text-[#2E8B57]" />}
            badge={<StatusBadge status={flaggedCount > 0 ? 'flagged' : 'accepted'} size="sm" />}
            onClick={() => onNavigate('data-logs')}
          />
        </div>
      )}

      {/* Model snapshot (prediction dependent) */}
      {overview &&
        (hasPredictions ? (
          <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center gap-x-6 gap-y-3 text-xs" data-testid="model-snapshot">
            <span className="font-bold uppercase tracking-wider text-[#6B786E]">Plants by ML severity (latest reading)</span>
            <div className="flex flex-wrap items-center gap-2">
              {(['healthy', 'low', 'moderate', 'high'] as const).map((s) => (
                <span key={s} className="inline-flex items-center gap-1.5">
                  <SeverityBadge severity={s} />
                  <strong className="font-mono text-[#1C2B23]">{overview.severityCounts[s] ?? 0}</strong>
                </span>
              ))}
            </div>
            <span className="text-stone-600">
              {formatCount(totals?.anomalies ?? 0)} unusual readings of {formatCount(totals?.predictions.ok ?? 0)} analyzed <AnomalyTip />
            </span>
            <span className="text-stone-500">Model {orNA(overview.modelVersion)}</span>
          </div>
        ) : (
          <RunEmptyState title="Model results not available yet" className="my-2 p-6" />
        ))}

      {/* Sensor Trends Section */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">Canopy &amp; Soil Environmental Oscillations</h2>
            <p className="text-xs text-[#6B786E]">
              One pot&apos;s moisture, temperature, humidity and light. Ranges are relative to the dataset&apos;s latest reading (the data is historical).
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              aria-label="Pot for trend chart"
              value={trendPlantId ?? ''}
              onChange={(e) => setTrendPlant(Number(e.target.value))}
              className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
            >
              {plants.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <TimeRangeButtons value={timeRange} onChange={setTimeRange} />
          </div>
        </div>

        {trendQuery.error ? (
          <ErrorState message={trendQuery.error.message} onRetry={() => void trendQuery.refetch()} />
        ) : trend ? (
          <SensorTrendChart
            records={trendSeries}
            title={`Telemetry Trends: ${trend.plant.name}`}
            optimalRange={trend.plant.optimalRanges}
            height={320}
          />
        ) : (
          <div className="bg-white rounded-card border border-[#DCE4D9] p-8 text-center text-xs text-stone-500">
            {plantsQuery.loading || trendQuery.loading ? 'Loading trend...' : 'No trend data available.'}
          </div>
        )}
      </div>

      {/* Alerts + recommendations (prediction dependent) */}
      {overview && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden" data-testid="recent-alerts">
            <div className="p-4 border-b border-[#DCE4D9]">
              <h3 className="text-sm font-bold text-[#1C2B23]">Recent Alerts</h3>
              <p className="text-xs text-[#6B786E]">Unusual readings and new care recommendations.</p>
            </div>
            {overview.recentAlerts.length === 0 ? (
              hasPredictions ? (
                <p className="p-6 text-center text-xs text-stone-500">No alerts.</p>
              ) : (
                <p className="p-6 text-center text-xs text-stone-500">{DISCLAIMERS.runPrompt}</p>
              )
            ) : (
              <ul className="divide-y divide-[#DCE4D9]/60">
                {overview.recentAlerts.map((a, i) => (
                  <li
                    key={`${a.kind}-${a.recordId ?? a.recommendationId ?? i}`}
                    onClick={() => onNavigate('plant-detail', a.plantId)}
                    className="px-4 py-3 text-xs cursor-pointer hover:bg-[#EDF4EB]/40"
                  >
                    <div className="flex items-center justify-between gap-2 mb-0.5">
                      <span className="font-bold text-[#1C2B23]">{a.plantName}</span>
                      <span className="text-[10px] font-mono text-stone-500">{formatDateTime(a.timestamp)}</span>
                    </div>
                    <div className="flex items-start gap-2 text-stone-600">
                      <span
                        className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
                          a.kind === 'anomaly' ? 'border-amber-300 bg-amber-50 text-amber-900' : 'border-[#DDE8D8] bg-[#EDF4EB] text-[#2E8B57]'
                        }`}
                      >
                        {a.kind === 'anomaly' ? 'Unusual reading' : 'Recommendation'}
                      </span>
                      <span className="leading-snug">
                        {a.message} {a.kind === 'anomaly' && <AnomalyTip />}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="space-y-3" data-testid="top-recommendations">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-[#1C2B23]">Top Recommended Actions</h3>
                <p className="text-xs text-[#6B786E]">{DISCLAIMERS.recommendation}</p>
              </div>
              <button onClick={() => onNavigate('recommendations')} className="text-xs font-bold text-[#2E8B57] hover:underline flex items-center gap-1">
                <span>All actions</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
            {actions.error && (
              <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800 flex items-start justify-between gap-2">
                <span>{describeRecommendationError(actions.error, overview?.topRecommendations ?? [])}</span>
                <button onClick={actions.clearError} className="font-semibold underline">
                  Dismiss
                </button>
              </div>
            )}
            {overview.topRecommendations.length === 0 ? (
              hasPredictions ? (
                <p className="rounded-card border border-[#DCE4D9] bg-white p-6 text-center text-xs text-stone-500">No active recommendations.</p>
              ) : (
                <RunEmptyState title="No recommendations yet" className="my-2 p-6" />
              )
            ) : (
              <div className="space-y-4">
                {overview.topRecommendations.slice(0, 2).map((rec) => (
                  <RecommendationCard
                    key={rec.id}
                    recommendation={rec}
                    pending={actions.isPending(rec.id)}
                    onUpdateStatus={(id, status) => void actions.update(id, status)}
                    onViewPlant={(id) => onNavigate('plant-detail', id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recent readings table */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="p-5 border-b border-[#DCE4D9] flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">Recent Sensor Activity Log</h3>
            <p className="text-xs text-[#6B786E]">Latest stored readings in the dataset.</p>
          </div>
          <button onClick={() => onNavigate('data-logs')} className="text-xs font-bold text-[#2E8B57] hover:underline flex items-center gap-1">
            <span>Open Full Data Explorer</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">Timestamp (IST)</th>
                <th className="py-3 px-4 font-semibold">Plant / Pot</th>
                <th className="py-3 px-4 font-semibold">Group</th>
                <th className="py-3 px-4 font-semibold">Node / Point</th>
                <th className="py-3 px-4 font-semibold">Soil Moisture</th>
                <th className="py-3 px-4 font-semibold">Temperature</th>
                <th className="py-3 px-4 font-semibold">Humidity</th>
                <th className="py-3 px-4 font-semibold">Light</th>
                <th className="py-3 px-4 font-semibold">Status</th>
                <th className="py-3 px-4 font-semibold text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60">
              {recentRecords.length === 0 && (
                <tr>
                  <td colSpan={10} className="py-6 px-4 text-center text-stone-500">
                    {recentQuery.loading ? 'Loading readings...' : 'No readings stored.'}
                  </td>
                </tr>
              )}
              {recentRecords.map((r) => {
                const plant = plants.find((p) => p.id === r.plant_id);
                return (
                  <tr
                    key={r.record_id}
                    className="hover:bg-[#EDF4EB]/40 transition-colors cursor-pointer"
                    onClick={() => setSelectedRecord(r)}
                  >
                    <td className="py-3 px-4 font-mono text-[#1C2B23]">{formatDateTime(r.timestamp)}</td>
                    <td className="py-3 px-4 font-bold text-[#1C2B23]">
                      Pot #{r.plant_id}
                      {plant && <span className="block text-[10px] text-stone-500 font-normal">{plant.variety}</span>}
                    </td>
                    <td className="py-3 px-4 text-stone-600">{plant ? plant.cluster : 'n/a'}</td>
                    <td className="py-3 px-4 font-mono text-stone-700">
                      Node {r.node_id} • Point {orNA(r.sampling_point)}
                    </td>
                    <td className="py-3 px-4 font-bold text-[#1C2B23]">
                      <span className={plant && r.soil !== null && r.soil < plant.optimalRanges.soil.min ? 'text-[#C98227]' : 'text-[#2E8B57]'}>{formatPercent(r.soil)}</span>
                    </td>
                    <td className="py-3 px-4 text-[#1C2B23] font-medium">{formatTemp(r.temperature)}</td>
                    <td className="py-3 px-4 text-[#1C2B23] font-medium">{formatPercent(r.humidity)}</td>
                    <td className="py-3 px-4 text-[#1C2B23] font-medium">{formatLux(r.light)}</td>
                    <td className="py-3 px-4">
                      <span title={r.validation_error ?? undefined}>
                        <StatusBadge status={r.status} size="sm" />
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedRecord(r);
                        }}
                        className="px-2.5 py-1 text-xs rounded bg-[#EDF4EB] text-[#2E8B57] font-semibold hover:bg-[#DDE8D8] transition-colors"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <RecordDetailDrawer record={selectedRecord} onClose={() => setSelectedRecord(null)} />
    </div>
  );
};
