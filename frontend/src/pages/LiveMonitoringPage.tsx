import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RefreshCw,
  Clock,
  Radio,
  Filter,
  ChevronRight,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Plant, SensorRecord } from '../types';
import { SensorReadingCard } from '../components/shared/SensorReadingCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { LiveSparkline } from '../components/charts/LiveSparkline';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { formatRelativeTime, formatPercent, formatTemp, formatLux } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface LiveMonitoringPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const LiveMonitoringPage: React.FC<LiveMonitoringPageProps> = ({ onNavigate }) => {
  const [plants, setPlants] = useState<Plant[]>([]);
  const [liveRecords, setLiveRecords] = useState<SensorRecord[]>([]);
  const [selectedPlantId, setSelectedPlantId] = useState<string>('5'); // Pot 5 (Water Stress) default
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [isPolling, setIsPolling] = useState<boolean>(true);
  const [pollingInterval, setPollingInterval] = useState<number>(3000);
  const [lastHeartbeat, setLastHeartbeat] = useState<Date>(new Date());
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);
  const [flashRecordId, setFlashRecordId] = useState<string | null>(null);

  const pollTimerRef = useRef<any>(null);

  useEffect(() => {
    loadInitial();
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    if (isPolling) {
      pollTimerRef.current = setInterval(() => {
        handlePollStep();
      }, pollingInterval);
    }

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [isPolling, pollingInterval]);

  const loadInitial = async () => {
    const [p, r] = await Promise.all([
      apiClient.getPlants(),
      apiClient.getRecords({ limit: 40 }),
    ]);
    setPlants(p);
    setLiveRecords(r);
    setLastHeartbeat(new Date());
  };

  // The bundled dataset is historical, so "polling" re-reads the latest stored readings from the API.
  const handlePollStep = () => {
    void loadInitial().catch(() => undefined);
  };

  const activePlant = plants.find((p) => p.id === Number(selectedPlantId)) || plants[4] || plants[0];
  const activeReading = activePlant?.latestReading;

  // Filter live stream table
  const filteredRecords = liveRecords.filter((rec) => {
    if (selectedCluster !== 'all') {
      const pl = plants.find((p) => p.id === rec.plant_id);
      if (pl?.cluster !== selectedCluster) return false;
    }
    if (selectedPlantId && selectedPlantId !== 'all') {
      return rec.plant_id === Number(selectedPlantId);
    }
    return true;
  });

  const plantHistory = liveRecords
    .filter((r) => r.plant_id === activePlant?.id)
    .slice(0, 15)
    .reverse();

  const soilSparkline = plantHistory.map((r) => r.soil).filter((v): v is number => v !== null);
  const tempSparkline = plantHistory.map((r) => r.temperature).filter((v): v is number => v !== null);
  const humSparkline = plantHistory.map((r) => r.humidity).filter((v): v is number => v !== null);
  const lightSparkline = plantHistory.map((r) => r.light).filter((v): v is number => v !== null);

  return (
    <div className="space-y-6">
      {/* Top Stream Controller & Diagnostics */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-3 w-3 relative">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  isPolling ? 'bg-[#2E8B57]' : 'bg-stone-400'
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-3 w-3 ${
                  isPolling ? 'bg-[#2E8B57]' : 'bg-stone-400'
                }`}
              />
            </span>
            <div>
              <span className="text-xs font-bold text-[#1C2B23]">
                Jetson Stream Receiver: {isPolling ? 'Live Telemetry Stream Active' : 'Stream Paused'}
              </span>
              <div className="text-[11px] text-[#6B786E] flex items-center gap-2 font-mono">
                <span>Last packet: {formatRelativeTime(lastHeartbeat.toISOString())}</span>
                <span>•</span>
                <span>Port 5000 / HTTP Sequential Ingestion</span>
              </div>
            </div>
          </div>
        </div>

        {/* Polling controls */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs bg-[#EDF4EB] px-2.5 py-1.5 rounded-lg border border-[#DDE8D8]">
            <Clock className="w-3.5 h-3.5 text-[#2E8B57]" />
            <span className="text-[#173F32] font-semibold">Speed:</span>
            <select
              value={pollingInterval}
              onChange={(e) => setPollingInterval(Number(e.target.value))}
              className="bg-transparent font-bold text-[#102F26] outline-none cursor-pointer"
            >
              <option value={2000}>2 seconds</option>
              <option value={3000}>3 seconds</option>
              <option value={5000}>5 seconds</option>
              <option value={10000}>10 seconds</option>
            </select>
          </div>

          <button
            onClick={() => setIsPolling(!isPolling)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm ${
              isPolling
                ? 'bg-amber-100 hover:bg-amber-200 text-amber-900 border border-amber-300'
                : 'bg-[#2E8B57] hover:bg-[#247046] text-white'
            }`}
          >
            {isPolling ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPolling ? 'Pause Stream' : 'Resume Stream'}</span>
          </button>

          <button
            onClick={handlePollStep}
            className="p-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 transition-colors"
            title="Step next packet manually"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Filter and Plant Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[#EDF4EB]/70 p-3 rounded-xl border border-[#DDE8D8]">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-[#173F32] font-semibold">
            <Filter className="w-3.5 h-3.5 text-[#2E8B57]" />
            <span>Select Focus Pot:</span>
          </div>
          <select
            value={selectedPlantId}
            onChange={(e) => setSelectedPlantId(e.target.value)}
            className="text-xs font-bold bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Pots (Aggregated Feed)</option>
            {plants.map((p) => (
              <option key={p.id} value={p.id}>
                Pot #{p.id}: {p.name} ({p.cluster})
              </option>
            ))}
          </select>

          <select
            value={selectedCluster}
            onChange={(e) => setSelectedCluster(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Clusters (4)</option>
            <option value="Cluster 1">Cluster 1: Healthy Control (Pots 1–4)</option>
            <option value="Cluster 2">Cluster 2: Water Stress (Pots 5–8)</option>
            <option value="Cluster 3">Cluster 3: Light Stress (Pots 9–12)</option>
            <option value="Cluster 4">Cluster 4: Temperature/Humidity Stress (Pots 13–16)</option>
          </select>
        </div>

        {activePlant && selectedPlantId !== 'all' && (
          <button
            onClick={() => onNavigate('plant-detail', activePlant.id)}
            className="text-xs font-bold text-[#2E8B57] hover:underline flex items-center gap-1"
          >
            <span>Open Pot #{activePlant.id} Dossier</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Active Focus Specimen Cards */}
      {activePlant && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-[#1C2B23] flex items-center gap-2">
                <span>{activePlant.name}</span>
                <span className="text-xs font-mono font-normal text-[#6B786E]">
                  (Node {activePlant.nodeId} • Point {activePlant.samplingPoint})
                </span>
                <StatusBadge status={activePlant.stressType} size="sm" />
              </h2>
              <div className="text-xs text-[#6B786E] flex items-center gap-3 mt-0.5">
                <span>Location: {activePlant.location}</span>
                <span>•</span>
                <span>Timestamp: {activeReading ? activeReading.timestamp : 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* 4 Sensor Cards with Sparklines */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <SensorReadingCard
                type="soil"
                value={activeReading?.soil}
                optimalRange={activePlant.optimalRanges.soil}
                timestamp={activeReading ? formatRelativeTime(activeReading.timestamp) : undefined}
              />
              <div className="bg-white p-2.5 rounded-xl border border-[#DCE4D9] text-[10px] text-[#6B786E] flex items-center justify-between shadow-sm">
                <span className="font-semibold">Recent Trend</span>
                <LiveSparkline data={soilSparkline} color="#2E8B57" height={22} width={110} />
              </div>
            </div>

            <div className="space-y-2">
              <SensorReadingCard
                type="temperature"
                value={activeReading?.temperature}
                optimalRange={activePlant.optimalRanges.temperature}
                timestamp={activeReading ? formatRelativeTime(activeReading.timestamp) : undefined}
              />
              <div className="bg-white p-2.5 rounded-xl border border-[#DCE4D9] text-[10px] text-[#6B786E] flex items-center justify-between shadow-sm">
                <span className="font-semibold">Recent Trend</span>
                <LiveSparkline data={tempSparkline} color="#C98227" height={22} width={110} />
              </div>
            </div>

            <div className="space-y-2">
              <SensorReadingCard
                type="humidity"
                value={activeReading?.humidity}
                optimalRange={activePlant.optimalRanges.humidity}
                timestamp={activeReading ? formatRelativeTime(activeReading.timestamp) : undefined}
              />
              <div className="bg-white p-2.5 rounded-xl border border-[#DCE4D9] text-[10px] text-[#6B786E] flex items-center justify-between shadow-sm">
                <span className="font-semibold">Recent Trend</span>
                <LiveSparkline data={humSparkline} color="#3977B8" height={22} width={110} />
              </div>
            </div>

            <div className="space-y-2">
              <SensorReadingCard
                type="light"
                value={activeReading?.light}
                optimalRange={activePlant.optimalRanges.light}
                timestamp={activeReading ? formatRelativeTime(activeReading.timestamp) : undefined}
              />
              <div className="bg-white p-2.5 rounded-xl border border-[#DCE4D9] text-[10px] text-[#6B786E] flex items-center justify-between shadow-sm">
                <span className="font-semibold">Recent Trend</span>
                <LiveSparkline data={lightSparkline} color="#D4A338" height={22} width={110} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Live Ingestion Feed Table */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="p-4 border-b border-[#DCE4D9] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
              <Radio className="w-4 h-4 text-[#2E8B57]" />
              <span>Real-Time Stream Feed</span>
            </h3>
            <p className="text-xs text-[#6B786E]">
              Sequential sensor packets arriving from Node 1 (Points 1–12) and Node 2 (Points 1–4).
            </p>
          </div>
          <span className="text-xs font-mono font-medium text-stone-500">
            {filteredRecords.length} records buffered
          </span>
        </div>

        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-xs text-left">
            <thead className="sticky top-0 bg-[#F6F7F0] border-b border-[#DCE4D9]">
              <tr className="text-[#6B786E]">
                <th className="py-2.5 px-4 font-semibold">timestamp</th>
                <th className="py-2.5 px-4 font-semibold">session_id</th>
                <th className="py-2.5 px-4 font-semibold">node_id</th>
                <th className="py-2.5 px-4 font-semibold">sampling_point</th>
                <th className="py-2.5 px-4 font-semibold">plant_id</th>
                <th className="py-2.5 px-4 font-semibold">soil (%)</th>
                <th className="py-2.5 px-4 font-semibold">temp (°C)</th>
                <th className="py-2.5 px-4 font-semibold">humidity (%)</th>
                <th className="py-2.5 px-4 font-semibold">light (lux)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60 font-mono">
              {filteredRecords.map((r, idx) => {
                const isFlash = r.record_id === flashRecordId || (r.plant_id === Number(flashRecordId));
                return (
                  <tr
                    key={r.record_id || idx}
                    onClick={() => setSelectedRecord(r)}
                    className={`cursor-pointer transition-colors ${
                      isFlash
                        ? 'bg-emerald-100 font-bold'
                        : 'hover:bg-[#EDF4EB]/40'
                    }`}
                  >
                    <td className="py-2.5 px-4 text-[#1C2B23]">
                      {r.timestamp}
                    </td>
                    <td className="py-2.5 px-4 text-stone-600">
                      {r.session_id}
                    </td>
                    <td className="py-2.5 px-4 font-bold text-stone-800">
                      "{r.node_id}"
                    </td>
                    <td className="py-2.5 px-4 text-stone-600">
                      {r.sampling_point}
                    </td>
                    <td className="py-2.5 px-4 font-bold text-[#102F26]">
                      Pot #{r.plant_id}
                    </td>
                    <td className="py-2.5 px-4 font-bold text-[#2E8B57]">
                      {formatPercent(r.soil)}
                    </td>
                    <td className="py-2.5 px-4 text-[#1C2B23]">
                      {formatTemp(r.temperature)}
                    </td>
                    <td className="py-2.5 px-4 text-[#1C2B23]">
                      {formatPercent(r.humidity)}
                    </td>
                    <td className="py-2.5 px-4 text-[#1C2B23]">
                      {formatLux(r.light)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <RecordDetailDrawer
        record={selectedRecord}
        onClose={() => setSelectedRecord(null)}
      />
    </div>
  );
};
