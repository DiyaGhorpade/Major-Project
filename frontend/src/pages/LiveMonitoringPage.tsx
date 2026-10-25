import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, Pause, RefreshCw, Clock, Radio, Filter, ChevronRight } from 'lucide-react';
import { apiClient } from '../api/client';
import { SensorRecord } from '../types';
import { useAsync } from '../hooks/useApi';
import { usePlants, usePlantMonitoring } from '../hooks/usePlants';
import { useSettings } from '../hooks/useSettings';
import { SensorReadingCard } from '../components/shared/SensorReadingCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { ErrorState } from '../components/shared/ErrorState';
import { LiveSparkline } from '../components/charts/LiveSparkline';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { PredictionSummary } from '../components/monitoring/PredictionBadges';
import { formatDateTime, formatPercent, formatTemp, formatLux, orNA } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface LiveMonitoringPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

const DEFAULT_POLL_SECONDS = 5;
const FEED_LIMIT = 40;
const SPARK_POINTS = 24;

export const LiveMonitoringPage: React.FC<LiveMonitoringPageProps> = ({ onNavigate }) => {
  const [selectedPlant, setSelectedPlant] = useState<string>('all'); // feed filter: "all" or a pot id
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [focusPlant, setFocusPlant] = useState<number | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(true);
  const [fetchedAt, setFetchedAt] = useState<Date | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);

  const plantsQuery = usePlants();
  const settingsQuery = useSettings();
  const plants = useMemo(() => plantsQuery.data ?? [], [plantsQuery.data]);
  const pollSeconds = settingsQuery.data?.polling_interval_seconds ?? DEFAULT_POLL_SECONDS;

  const clusters = useMemo(() => {
    const seen = new Map<string, string>();
    for (const p of plants) if (!seen.has(p.cluster)) seen.set(p.cluster, p.clusterLabel || p.cluster);
    return [...seen.entries()];
  }, [plants]);

  const visiblePlants = plants.filter((p) => selectedCluster === 'all' || p.cluster === selectedCluster);
  // Focus pot: explicit pick (if still visible), else the filtered pot, else the first visible pot. No hard-coded default.
  const focusId =
    selectedPlant !== 'all'
      ? Number(selectedPlant)
      : focusPlant !== null && visiblePlants.some((p) => p.id === focusPlant)
        ? focusPlant
        : (visiblePlants[0]?.id ?? null);

  const feedQuery = useAsync(
    async () => {
      const page = await apiClient.getRecordsPage({
        limit: FEED_LIMIT,
        plantId: selectedPlant === 'all' ? undefined : Number(selectedPlant),
        cluster: selectedCluster === 'all' ? undefined : selectedCluster,
      });
      setFetchedAt(new Date());
      return page;
    },
    ['live-feed', selectedPlant, selectedCluster],
  );
  const monitoringQuery = usePlantMonitoring(focusId, SPARK_POINTS);

  // The bundled dataset is historical, so "polling" simply re-reads the latest stored readings.
  const refetchers = useRef<Array<() => Promise<void>>>([]);
  useEffect(() => {
    refetchers.current = [feedQuery.refetch, monitoringQuery.refetch];
  });
  const refreshNow = useCallback(() => {
    for (const fn of refetchers.current) void fn();
  }, []);
  useEffect(() => {
    if (!isPolling) return;
    const id = setInterval(refreshNow, pollSeconds * 1000);
    return () => clearInterval(id);
  }, [isPolling, pollSeconds, refreshNow]);

  const feed = feedQuery.data?.items ?? [];
  const monitoring = monitoringQuery.data;
  const activePlant = monitoring?.plant ?? null;
  const activeReading = activePlant?.latestReading ?? null;
  const series = monitoring?.series ?? [];
  const loadError = plantsQuery.error ?? feedQuery.error ?? monitoringQuery.error;

  return (
    <div className="space-y-6">
      {loadError && <ErrorState message={loadError.message} onRetry={refreshNow} />}

      {/* Honest status header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-3 w-3 relative">
            <span className={`relative inline-flex rounded-full h-3 w-3 ${isPolling ? 'bg-[#2E8B57]' : 'bg-stone-400'}`} />
          </span>
          <div>
            <span className="text-xs font-bold text-[#1C2B23]">
              Showing the bundled dataset&apos;s most recent readings
            </span>
            <div className="text-[11px] text-[#6B786E] flex flex-wrap items-center gap-2">
              <span>No live hardware is connected: this view re-reads stored data.</span>
              <span>•</span>
              <span className="font-mono">
                {isPolling ? `Auto-refresh every ${pollSeconds} s` : 'Auto-refresh paused'}
                {fetchedAt ? ` • last refreshed ${formatDateTime(fetchedAt.toISOString())} IST` : ''}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs bg-[#EDF4EB] px-2.5 py-1.5 rounded-lg border border-[#DDE8D8]" title="Change in Settings">
            <Clock className="w-3.5 h-3.5 text-[#2E8B57]" />
            <span className="text-[#173F32] font-semibold">Interval:</span>
            <span className="font-bold text-[#102F26]">{pollSeconds} s</span>
          </div>

          <button
            onClick={() => setIsPolling(!isPolling)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm ${
              isPolling ? 'bg-amber-100 hover:bg-amber-200 text-amber-900 border border-amber-300' : 'bg-[#2E8B57] hover:bg-[#247046] text-white'
            }`}
          >
            {isPolling ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPolling ? 'Pause Refresh' : 'Resume Refresh'}</span>
          </button>

          <button
            onClick={refreshNow}
            aria-label="Refresh now"
            className="p-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 transition-colors"
            title="Refresh now"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${feedQuery.refreshing ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[#EDF4EB]/70 p-3 rounded-xl border border-[#DDE8D8]">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-[#173F32] font-semibold">
            <Filter className="w-3.5 h-3.5 text-[#2E8B57]" />
            <span>Pot:</span>
          </div>
          <select
            aria-label="Pot filter"
            value={selectedPlant}
            onChange={(e) => setSelectedPlant(e.target.value)}
            className="text-xs font-bold bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All pots (feed)</option>
            {visiblePlants.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.cluster})
              </option>
            ))}
          </select>

          <select
            aria-label="Group filter"
            value={selectedCluster}
            onChange={(e) => {
              setSelectedCluster(e.target.value);
              setSelectedPlant('all');
            }}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All groups ({clusters.length})</option>
            {clusters.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>

          {selectedPlant === 'all' && visiblePlants.length > 0 && (
            <select
              aria-label="Focus pot"
              value={focusId ?? ''}
              onChange={(e) => setFocusPlant(Number(e.target.value))}
              className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
            >
              {visiblePlants.map((p) => (
                <option key={p.id} value={p.id}>
                  Focus: Pot #{p.id}
                </option>
              ))}
            </select>
          )}
        </div>

        {activePlant && (
          <button
            onClick={() => onNavigate('plant-detail', activePlant.id)}
            className="text-xs font-bold text-[#2E8B57] hover:underline flex items-center gap-1"
          >
            <span>Open Pot #{activePlant.id} Details</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {!plantsQuery.loading && plants.length === 0 && !plantsQuery.error && (
        <div className="rounded-card border border-[#DCE4D9] bg-white p-8 text-center text-xs text-stone-500">No plants found.</div>
      )}

      {/* Focus pot cards */}
      {activePlant && (
        <div className="space-y-3" data-testid="focus-plant">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23] flex flex-wrap items-center gap-2">
              <span>{activePlant.name}</span>
              <span className="text-xs font-mono font-normal text-[#6B786E]">
                (Node {activePlant.nodeId} • Point {orNA(activePlant.samplingPoint)})
              </span>
              <span className="rounded-full border border-[#DDE8D8] bg-[#EDF4EB] px-2 py-0.5 text-xs font-medium text-[#173F32]">{activePlant.clusterLabel}</span>
            </h2>
            <div className="text-xs text-[#6B786E] flex flex-wrap items-center gap-3 mt-0.5">
              <span>Location: {activePlant.location}</span>
              <span>•</span>
              <span>Reading time (IST): {activeReading ? formatDateTime(activeReading.timestamp) : 'n/a'}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {(
              [
                { type: 'soil', key: 'soil', color: '#2E8B57' },
                { type: 'temperature', key: 'temperature', color: '#C98227' },
                { type: 'humidity', key: 'humidity', color: '#3977B8' },
                { type: 'light', key: 'light', color: '#D4A338' },
              ] as const
            ).map((m) => (
              <div key={m.type} className="space-y-2">
                <SensorReadingCard
                  type={m.type}
                  value={activeReading?.[m.key]}
                  optimalRange={activePlant.optimalRanges[m.key]}
                  timestamp={activeReading ? formatDateTime(activeReading.timestamp) : undefined}
                  readingTime={activeReading?.timestamp}
                />
                <div className="bg-white p-2.5 rounded-xl border border-[#DCE4D9] text-[10px] text-[#6B786E] flex items-center justify-between shadow-sm">
                  <span className="font-semibold">Recent Trend</span>
                  <LiveSparkline data={series.map((s) => s[m.key])} color={m.color} height={22} width={110} />
                </div>
              </div>
            ))}
          </div>

          <PredictionSummary prediction={activePlant.conditionPrediction} className="max-w-md" />
        </div>
      )}
      {!activePlant && (monitoringQuery.loading || plantsQuery.loading) && (
        <div className="p-6 text-center text-xs text-stone-500">Loading latest readings...</div>
      )}

      {/* Feed table */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="p-4 border-b border-[#DCE4D9] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
              <Radio className="w-4 h-4 text-[#2E8B57]" />
              <span>Latest Stored Readings</span>
            </h3>
            <p className="text-xs text-[#6B786E]">Newest readings first, from the bundled dataset (not a live stream).</p>
          </div>
          <span className="text-xs font-mono font-medium text-stone-500">
            {feed.length} of {feedQuery.data?.total ?? 0} shown
          </span>
        </div>

        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-xs text-left">
            <thead className="sticky top-0 bg-[#F6F7F0] border-b border-[#DCE4D9]">
              <tr className="text-[#6B786E]">
                <th className="py-2.5 px-4 font-semibold">timestamp (IST)</th>
                <th className="py-2.5 px-4 font-semibold">session_id</th>
                <th className="py-2.5 px-4 font-semibold">node_id</th>
                <th className="py-2.5 px-4 font-semibold">sampling_point</th>
                <th className="py-2.5 px-4 font-semibold">plant_id</th>
                <th className="py-2.5 px-4 font-semibold">soil (%)</th>
                <th className="py-2.5 px-4 font-semibold">temp (°C)</th>
                <th className="py-2.5 px-4 font-semibold">humidity (%)</th>
                <th className="py-2.5 px-4 font-semibold">light (lux)</th>
                <th className="py-2.5 px-4 font-semibold">status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60 font-mono">
              {feed.length === 0 && (
                <tr>
                  <td colSpan={10} className="py-6 px-4 text-center text-stone-500 font-sans">
                    {feedQuery.loading ? 'Loading readings...' : 'No readings match these filters.'}
                  </td>
                </tr>
              )}
              {feed.map((r) => (
                <tr key={r.record_id} onClick={() => setSelectedRecord(r)} className="cursor-pointer transition-colors hover:bg-[#EDF4EB]/40">
                  <td className="py-2.5 px-4 text-[#1C2B23]">{formatDateTime(r.timestamp)}</td>
                  <td className="py-2.5 px-4 text-stone-600">{orNA(r.session_id)}</td>
                  <td className="py-2.5 px-4 font-bold text-stone-800">{r.node_id}</td>
                  <td className="py-2.5 px-4 text-stone-600">{orNA(r.sampling_point)}</td>
                  <td className="py-2.5 px-4 font-bold text-[#102F26]">Pot #{r.plant_id}</td>
                  <td className="py-2.5 px-4 font-bold text-[#2E8B57]">{formatPercent(r.soil)}</td>
                  <td className="py-2.5 px-4 text-[#1C2B23]">{formatTemp(r.temperature)}</td>
                  <td className="py-2.5 px-4 text-[#1C2B23]">{formatPercent(r.humidity)}</td>
                  <td className="py-2.5 px-4 text-[#1C2B23]">{formatLux(r.light)}</td>
                  <td className="py-2.5 px-4 font-sans">
                    <span title={r.validation_error ?? undefined}>
                      <StatusBadge status={r.status} size="sm" />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <RecordDetailDrawer record={selectedRecord} onClose={() => setSelectedRecord(null)} />
    </div>
  );
};
