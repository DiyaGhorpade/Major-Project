import React, { useMemo, useState } from 'react';
import {
  LayoutGrid,
  List,
  Search,
  Droplets,
  Thermometer,
  Wind,
  Sun,
  ChevronRight,
} from 'lucide-react';
import { usePlants } from '../hooks/usePlants';
import { ErrorState } from '../components/shared/ErrorState';
import { PlantStatusBadge, PredictionSummary, SeverityBadge, AnomalyFlag, EnvStateLabel } from '../components/monitoring/PredictionBadges';
import { DISCLAIMERS } from '../components/shared/disclaimerText';
import { formatConfidence } from '../components/monitoring/format';
import { formatDateTime, formatPercent, formatTemp, formatLux, orNA } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface PlantsPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const PlantsPage: React.FC<PlantsPageProps> = ({ onNavigate }) => {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [selectedNode, setSelectedNode] = useState<string>('all');
  const plantsQuery = usePlants();
  const plants = useMemo(() => plantsQuery.data ?? [], [plantsQuery.data]);

  // Filter options come from the data (group labels, nodes and the pots they read).
  const clusterOptions = useMemo(() => {
    const seen = new Map<string, string>();
    for (const p of plants) if (!seen.has(p.cluster)) seen.set(p.cluster, p.clusterLabel || p.cluster);
    return [...seen.entries()];
  }, [plants]);
  const nodeOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of plants) counts.set(p.nodeId, (counts.get(p.nodeId) ?? 0) + 1);
    return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [plants]);

  const filteredPlants = plants.filter((plant) => {
    if (selectedCluster !== 'all' && plant.cluster !== selectedCluster) return false;
    if (selectedNode !== 'all' && plant.nodeId !== selectedNode) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchName = plant.name.toLowerCase().includes(q);
      const matchId = String(plant.id).includes(q);
      const matchSpecies = plant.species.toLowerCase().includes(q);
      const matchCluster = plant.cluster.toLowerCase().includes(q);
      const matchStress = plant.stressType.toLowerCase().includes(q);
      return matchName || matchId || matchSpecies || matchCluster || matchStress;
    }
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Controls & Search Filter Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 flex-1">
          {/* Search */}
          <div className="relative min-w-[240px] flex-1 max-w-sm">
            <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by Pot #, Rama/Krishna, or group..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full text-xs bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg pl-9 pr-3 py-2 outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
            />
          </div>

          {/* Cluster filter */}
          <select
            value={selectedCluster}
            onChange={(e) => setSelectedCluster(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-2 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All experimental groups ({clusterOptions.length})</option>
            {clusterOptions.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>

          {/* Node filter */}
          <select
            value={selectedNode}
            onChange={(e) => setSelectedNode(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-2 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All sensor nodes</option>
            {nodeOptions.map(([id, count]) => (
              <option key={id} value={id}>
                Node {id} ({count} pots)
              </option>
            ))}
          </select>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-1 bg-[#EDF4EB] p-1 rounded-lg border border-[#DDE8D8]">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-1.5 rounded-md transition-colors ${
              viewMode === 'grid'
                ? 'bg-white text-[#102F26] shadow-sm'
                : 'text-[#6B786E] hover:text-[#102F26]'
            }`}
            title="Card Grid View"
          >
            <LayoutGrid className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-1.5 rounded-md transition-colors ${
              viewMode === 'list'
                ? 'bg-white text-[#102F26] shadow-sm'
                : 'text-[#6B786E] hover:text-[#102F26]'
            }`}
            title="Dense List View"
          >
            <List className="w-4 h-4" />
          </button>
        </div>
      </div>

      {plantsQuery.error && <ErrorState message={plantsQuery.error.message} onRetry={() => void plantsQuery.refetch()} />}
      {plantsQuery.loading && <div className="p-6 text-center text-xs text-stone-500">Loading plants...</div>}
      {!plantsQuery.loading && !plantsQuery.error && filteredPlants.length === 0 && (
        <div className="rounded-card border border-[#DCE4D9] bg-white p-8 text-center text-xs text-stone-500">
          {plants.length === 0 ? 'No plants found.' : 'No pots match these filters.'}
        </div>
      )}
      {!plantsQuery.loading && plants.length > 0 && !plants.some((p) => p.conditionPrediction) && (
        <div className="rounded-lg border border-dashed border-stone-300 bg-white px-4 py-2.5 text-xs text-stone-600">
          Not analyzed yet. {DISCLAIMERS.runPrompt}
        </div>
      )}

      {/* Grid Mode */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {filteredPlants.map((plant) => {
            const reading = plant.latestReading;

            return (
              <div
                key={plant.id}
                data-testid={`plant-card-${plant.id}`}
                onClick={() => onNavigate('plant-detail', plant.id)}
                className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:shadow-card-hover hover:border-[#2E8B57] transition-all cursor-pointer flex flex-col justify-between group"
              >
                <div>
                  {/* Card header: pot badge, experimental group, model status */}
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="w-8 h-8 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-extrabold text-xs shadow-sm">
                        #{plant.id}
                      </span>
                      <div>
                        <span
                          className="text-[10px] font-bold uppercase tracking-wider text-[#6B786E]"
                          title="Experimental group (designed treatment), not a model cluster"
                        >
                          {plant.clusterLabel || plant.cluster}
                        </span>
                        <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors leading-tight">
                          {plant.name}
                        </h3>
                      </div>
                    </div>
                    <PlantStatusBadge status={plant.status} />
                  </div>

                  <p className="text-[11px] text-stone-500 italic mb-3">
                    {plant.variety} • {plant.species}
                  </p>

                  {/* Node & point mapping (from the data) */}
                  <div className="flex items-center justify-between text-[11px] text-stone-600 bg-stone-50 px-2.5 py-1.5 rounded-lg mb-3 border border-stone-200/60 font-mono">
                    <span>Node {plant.nodeId}</span>
                    <span>Point {orNA(plant.samplingPoint)}</span>
                  </div>

                  {/* Sensor metrics snapshot */}
                  {reading ? (
                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                      <div className="p-2 rounded-lg bg-[#EDF4EB]/70 border border-[#DDE8D8]">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Droplets className="w-3 h-3 text-[#2E8B57]" /> Soil
                        </div>
                        <span className="font-bold text-[#1C2B23]">{formatPercent(reading.soil)}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-stone-50 border border-stone-200/60">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Thermometer className="w-3 h-3 text-amber-600" /> Temp
                        </div>
                        <span className="font-bold text-[#1C2B23]">{formatTemp(reading.temperature)}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-stone-50 border border-stone-200/60">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Wind className="w-3 h-3 text-sky-600" /> Humidity
                        </div>
                        <span className="font-bold text-[#1C2B23]">{formatPercent(reading.humidity)}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-stone-50 border border-stone-200/60">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Sun className="w-3 h-3 text-yellow-600" /> Light
                        </div>
                        <span className="font-bold text-[#1C2B23]">{formatLux(reading.light)}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="mb-3 rounded-lg border border-dashed border-stone-300 px-2.5 py-2 text-[11px] text-stone-500">No reading stored.</div>
                  )}

                  <PredictionSummary prediction={plant.conditionPrediction} className="mb-3" />

                  {/* Simple Care Tip Box */}
                  <div className="p-2 rounded-lg bg-stone-50 border border-stone-200 text-[11px] text-stone-700 mb-3">
                    {plant.simpleCareTip}
                  </div>
                </div>

                {/* Footer */}
                <div className="pt-3 border-t border-[#DCE4D9] flex items-center justify-between text-[11px] text-stone-500">
                  <span>{reading ? formatDateTime(reading.timestamp) : 'No reading'}</span>
                  <span className="text-[#2E8B57] font-bold flex items-center gap-0.5 group-hover:translate-x-0.5 transition-transform">
                    Inspect Pot <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* List Mode */
        <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                  <th className="py-3 px-4 font-semibold">Pot # & Variety</th>
                  <th className="py-3 px-4 font-semibold">Group</th>
                  <th className="py-3 px-4 font-semibold">Node / Channel</th>
                  <th className="py-3 px-4 font-semibold">Soil (%)</th>
                  <th className="py-3 px-4 font-semibold">Temp (°C)</th>
                  <th className="py-3 px-4 font-semibold">Humidity (%)</th>
                  <th className="py-3 px-4 font-semibold">Light (lux)</th>
                  <th className="py-3 px-4 font-semibold">Model status</th>
                  <th className="py-3 px-4 font-semibold">
                    <span className="inline-flex items-center gap-1">
                      Stress type guess
                    </span>
                  </th>
                  <th className="py-3 px-4 font-semibold">Environmental state</th>
                  <th className="py-3 px-4 font-semibold">Reading check</th>
                  <th className="py-3 px-4 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DCE4D9]/60">
                {filteredPlants.map((plant) => {
                  const r = plant.latestReading;
                  const pred = plant.conditionPrediction;
                  return (
                    <tr
                      key={plant.id}
                      onClick={() => onNavigate('plant-detail', plant.id)}
                      className="hover:bg-[#EDF4EB]/40 transition-colors cursor-pointer"
                    >
                      <td className="py-3 px-4">
                        <div className="font-bold text-[#1C2B23]">{plant.name}</div>
                        <div className="text-[10px] text-stone-500 italic">{plant.species}</div>
                      </td>
                      <td className="py-3 px-4 text-stone-600">{plant.cluster}</td>
                      <td className="py-3 px-4 font-mono text-stone-600">
                        Node {plant.nodeId} • Point {orNA(plant.samplingPoint)}
                      </td>
                      <td className="py-3 px-4 font-bold text-[#2E8B57]">{r ? formatPercent(r.soil) : 'n/a'}</td>
                      <td className="py-3 px-4 font-medium text-[#1C2B23]">{r ? formatTemp(r.temperature) : 'n/a'}</td>
                      <td className="py-3 px-4 font-medium text-[#1C2B23]">{r ? formatPercent(r.humidity) : 'n/a'}</td>
                      <td className="py-3 px-4 font-medium text-[#1C2B23]">{r ? formatLux(r.light) : 'n/a'}</td>
                      <td className="py-3 px-4">
                        <div className="flex flex-col items-start gap-1">
                          <PlantStatusBadge status={plant.status} />
                          {pred && (
                            <span className="flex items-center gap-1 text-[10px] text-stone-500">
                              <SeverityBadge severity={pred.severity} /> {formatConfidence(pred.confidence)}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-stone-700">{pred ? orNA(pred.condition) : 'n/a'}</td>
                      <td className="py-3 px-4 text-stone-700">{pred ? <EnvStateLabel envState={pred.envState} /> : 'n/a'}</td>
                      <td className="py-3 px-4">{pred ? <AnomalyFlag isAnomaly={pred.anomaly?.isAnomaly} /> : 'n/a'}</td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onNavigate('plant-detail', plant.id);
                          }}
                          className="px-2.5 py-1 text-xs rounded bg-[#EDF4EB] text-[#2E8B57] font-semibold hover:bg-[#DDE8D8] transition-colors"
                        >
                          Details
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
