import React, { useState } from 'react';
import {
  Cpu,
  Wifi,
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
} from 'lucide-react';
import { SensorRecord } from '../types';
import { usePlants } from '../hooks/usePlants';
import { useReadings } from '../hooks/useReadings';
import { useDevices, useOverview, useSessions } from '../hooks/useData';
import { ErrorState } from '../components/shared/ErrorState';
import { MetricCard } from '../components/shared/MetricCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { SensorTrendChart } from '../components/charts/SensorTrendChart';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { formatRelativeTime, formatPercent, formatTemp, formatLux } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface OverviewPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate }) => {
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);
  const [timeRange, setTimeRange] = useState<'24h' | '48h' | '7d'>('48h');

  const plantsQuery = usePlants();
  const readingsQuery = useReadings({ limit: 100 });
  const devicesQuery = useDevices();
  const sessionsQuery = useSessions();
  const overviewQuery = useOverview();
  const plants = plantsQuery.data ?? [];
  const records = readingsQuery.data ?? [];
  const devices = devicesQuery.data ?? [];
  const sessions = sessionsQuery.data ?? [];
  const loadError = plantsQuery.error ?? readingsQuery.error ?? devicesQuery.error ?? sessionsQuery.error ?? overviewQuery.error;

  const jetson = devices.find((d) => d.type === 'jetson') || devices[0];
  const espNodes = devices.filter((d) => d.type === 'esp32');
  const activeSession = sessions[0];

  const totals = overviewQuery.data?.totals;
  const totalTransmissions = totals?.readings ?? records.length;
  const flaggedCount = totals?.flaggedReadings ?? records.filter((r) => r.status === 'flagged').length;
  const acceptanceRate =
    totalTransmissions > 0 ? (((totalTransmissions - flaggedCount) / totalTransmissions) * 100).toFixed(1) : '0.0';

  const recentRecords = records.slice(0, 8);

  // Per-group summary from real plant statuses (nothing is shown as "thriving" before the model has run).
  const clusterBadge = (cluster: string) => {
    const group = plants.filter((p) => p.cluster === cluster);
    if (group.length === 0 || !group.some((p) => p.conditionPrediction)) return 'Not analyzed yet';
    return `${group.filter((p) => p.status === 'healthy').length}/${group.length} rated healthy`;
  };

  return (
    <div className="space-y-6">
      {loadError && <ErrorState message={loadError.message} />}

      {/* Top Banner */}
      <div className="bg-gradient-to-r from-[#102F26] to-[#173F32] rounded-card p-6 text-white shadow-lg flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#79B86A] animate-pulse" />
            <span className="text-xs font-bold uppercase tracking-wider text-[#DDE8D8]">
              Bundled Dataset Replay
            </span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold tracking-tight">
            Tulsi Phenotype Stress Telemetry Dashboard
          </h1>
          <p className="text-xs text-[#DDE8D8]/80 mt-1 max-w-2xl leading-relaxed">
            16 monitored Tulsi pots (*Ocimum sanctum* & *tenuiflorum*) across 4 stress trial groups. Node 1 reads the odd-numbered pots; Node 2 reads the even-numbered pots.
          </p>
        </div>

        {/* Quick jump actions */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => onNavigate('live-monitoring')}
            className="px-3.5 py-2 rounded-lg bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Live Stream</span>
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

      {/* 4 Stress Cluster Visual Status Cues (Layman Friendly!) */}
      <div>
        <h2 className="text-xs font-bold uppercase tracking-wider text-[#6B786E] mb-3">
          At-A-Glance: 16 Tulsi Pots Across 4 Stress Trial Clusters
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Cluster 1: Healthy */}
          <div
            onClick={() => onNavigate('plants')}
            className="p-4 rounded-xl bg-white border border-[#DCE4D9] shadow-card hover:border-[#2E8B57] cursor-pointer transition-all flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-bold">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-bold text-[#1C2B23] block">Cluster 1: Healthy</span>
                <span className="text-[11px] text-[#6B786E]">Pots 1 to 4 (Control Group)</span>
              </div>
            </div>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] border border-[#DDE8D8]">
              {clusterBadge('Cluster 1')}
            </span>
          </div>

          {/* Cluster 2: Water Stress */}
          <div
            onClick={() => onNavigate('plants')}
            className="p-4 rounded-xl bg-white border border-amber-200 shadow-card hover:border-amber-400 cursor-pointer transition-all flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center font-bold">
                <Droplets className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <span className="text-xs font-bold text-amber-950 block">Cluster 2: Water Stress</span>
                <span className="text-[11px] text-[#6B786E]">Pots 5 to 8 (Drought Trial)</span>
              </div>
            </div>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-300">
              {clusterBadge('Cluster 2')}
            </span>
          </div>

          {/* Cluster 3: Light Stress */}
          <div
            onClick={() => onNavigate('plants')}
            className="p-4 rounded-xl bg-white border border-yellow-200 shadow-card hover:border-yellow-400 cursor-pointer transition-all flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-yellow-50 text-yellow-700 flex items-center justify-center font-bold">
                <Sun className="w-5 h-5 text-yellow-600" />
              </div>
              <div>
                <span className="text-xs font-bold text-yellow-950 block">Cluster 3: Light Stress</span>
                <span className="text-[11px] text-[#6B786E]">Pots 9 to 12 (Shade Box)</span>
              </div>
            </div>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-900 border border-yellow-300">
              {clusterBadge('Cluster 3')}
            </span>
          </div>

          {/* Cluster 4: Temperature/Humidity Stress */}
          <div
            onClick={() => onNavigate('plants')}
            className="p-4 rounded-xl bg-white border border-purple-200 shadow-card hover:border-purple-400 cursor-pointer transition-all flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-purple-50 text-purple-700 flex items-center justify-center font-bold">
                <Sprout className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <span className="text-xs font-bold text-purple-950 block">Cluster 4: Temperature/Humidity Stress</span>
                <span className="text-[11px] text-[#6B786E]">Pots 13 to 16 (Climate Trial)</span>
              </div>
            </div>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-purple-100 text-purple-900 border border-purple-300">
              {clusterBadge('Cluster 4')}
            </span>
          </div>
        </div>
      </div>

      {/* Operational Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Jetson Gateway"
          value={jetson ? jetson.status.charAt(0).toUpperCase() + jetson.status.slice(1) : 'Unknown'}
          subtitle={jetson?.last_heartbeat ? `Heartbeat: ${formatRelativeTime(jetson.last_heartbeat)}` : 'No live heartbeat (historical dataset)'}
          icon={<Cpu className="w-4 h-4 text-[#2E8B57]" />}
          badge={<StatusBadge status={jetson?.status ?? 'unknown'} size="sm" />}
          onClick={() => onNavigate('devices-network')}
        />

        <MetricCard
          title="ESP32 Nodes Configured"
          value={`${espNodes.filter((n) => n.status === 'online').length} / ${espNodes.length} Online`}
          subtitle="Node 1 (odd pots) • Node 2 (even pots)"
          icon={<Wifi className="w-4 h-4 text-[#2E8B57]" />}
          trend={{ value: '16 Pots Mapped', direction: 'up' }}
          onClick={() => onNavigate('devices-network')}
        />

        <MetricCard
          title="Current Collection Pass"
          value={activeSession ? `Session #${activeSession.session_id}` : 'No sessions'}
          subtitle={activeSession ? `${activeSession.received_records} of ${activeSession.expected_records} readings received` : 'n/a'}
          icon={<History className="w-4 h-4 text-[#2E8B57]" />}
          badge={<StatusBadge status="completed" size="sm" />}
          onClick={() => onNavigate('sessions')}
        />

        <MetricCard
          title="Ingestion Integrity"
          value={`${acceptanceRate}% Accepted`}
          subtitle={`${totalTransmissions} records logged (${flaggedCount} flagged)`}
          icon={<ShieldCheck className="w-4 h-4 text-[#2E8B57]" />}
          badge={<StatusBadge status="accepted" size="sm" />}
          onClick={() => onNavigate('data-logs')}
        />
      </div>

      {/* Sensor Trends Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">
              Canopy & Soil Environmental Oscillations
            </h2>
            <p className="text-xs text-[#6B786E]">
              Compare moisture, temperature, humidity, and light intensity across collection passes.
            </p>
          </div>
          <div className="flex items-center gap-1 bg-white border border-[#DCE4D9] rounded-lg p-1 text-xs">
            <button
              onClick={() => setTimeRange('24h')}
              className={`px-2.5 py-1 rounded ${timeRange === '24h' ? 'bg-[#EDF4EB] text-[#2E8B57] font-bold' : 'text-[#6B786E]'}`}
            >
              24h
            </button>
            <button
              onClick={() => setTimeRange('48h')}
              className={`px-2.5 py-1 rounded ${timeRange === '48h' ? 'bg-[#EDF4EB] text-[#2E8B57] font-bold' : 'text-[#6B786E]'}`}
            >
              48h
            </button>
            <button
              onClick={() => setTimeRange('7d')}
              className={`px-2.5 py-1 rounded ${timeRange === '7d' ? 'bg-[#EDF4EB] text-[#2E8B57] font-bold' : 'text-[#6B786E]'}`}
            >
              7d
            </button>
          </div>
        </div>

        <SensorTrendChart
          records={records}
          title="Telemetry Trends Across Collection Sessions"
          height={320}
        />
      </div>

      {/* Recent Telemetry Ingest Table */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="p-5 border-b border-[#DCE4D9] flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              Recent Sensor Activity Log
            </h3>
            <p className="text-xs text-[#6B786E]">
              Latest stored readings from Node 1 and Node 2.
            </p>
          </div>
          <button
            onClick={() => onNavigate('data-logs')}
            className="text-xs font-bold text-[#2E8B57] hover:underline flex items-center gap-1"
          >
            <span>Open Full Data Explorer</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">Timestamp</th>
                <th className="py-3 px-4 font-semibold">Plant / Pot</th>
                <th className="py-3 px-4 font-semibold">Cluster</th>
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
              {recentRecords.map((r) => {
                const plant = plants.find((p) => p.id === r.plant_id);
                return (
                  <tr
                    key={r.record_id || `${r.timestamp}-${r.plant_id}`}
                    className="hover:bg-[#EDF4EB]/40 transition-colors cursor-pointer"
                    onClick={() => setSelectedRecord(r)}
                  >
                    <td className="py-3 px-4 font-mono text-[#1C2B23]">
                      {r.timestamp}
                    </td>
                    <td className="py-3 px-4 font-bold text-[#1C2B23]">
                      Pot #{r.plant_id}
                      {plant && (
                        <span className="block text-[10px] text-stone-500 font-normal">
                          {plant.variety}
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-stone-600">
                      {plant ? plant.cluster : `Cluster ${Math.ceil(r.plant_id / 4)}`}
                    </td>
                    <td className="py-3 px-4 font-mono text-stone-700">
                      Node {r.node_id} • Point {r.sampling_point}
                    </td>
                    <td className="py-3 px-4 font-bold text-[#1C2B23]">
                      <span className={r.soil !== null && r.soil < 20 ? 'text-[#C98227]' : 'text-[#2E8B57]'}>
                        {formatPercent(r.soil)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-[#1C2B23] font-medium">
                      {formatTemp(r.temperature)}
                    </td>
                    <td className="py-3 px-4 text-[#1C2B23] font-medium">
                      {formatPercent(r.humidity)}
                    </td>
                    <td className="py-3 px-4 text-[#1C2B23] font-medium">
                      {formatLux(r.light)}
                    </td>
                    <td className="py-3 px-4">
                      <StatusBadge status={r.status || 'accepted'} size="sm" />
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

      {/* Record Inspector Drawer */}
      <RecordDetailDrawer
        record={selectedRecord}
        onClose={() => setSelectedRecord(null)}
      />
    </div>
  );
};
