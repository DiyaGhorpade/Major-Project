import React, { useState, useEffect } from 'react';
import {
  LayoutGrid,
  List,
  Search,
  Filter,
  Droplets,
  Thermometer,
  Wind,
  Sun,
  ChevronRight,
  Sparkles,
  Sprout,
  CheckCircle2,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Plant, ClusterId } from '../types';
import { StatusBadge } from '../components/shared/StatusBadge';
import { formatRelativeTime } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface PlantsPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const PlantsPage: React.FC<PlantsPageProps> = ({ onNavigate }) => {
  const [plants, setPlants] = useState<Plant[]>([]);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [selectedNode, setSelectedNode] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPlants();
  }, []);

  const loadPlants = async () => {
    setLoading(true);
    try {
      const p = await apiClient.getPlants();
      setPlants(p);
    } finally {
      setLoading(false);
    }
  };

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
              placeholder="Search by Pot #, Rama/Krishna, or stress condition..."
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
            <option value="all">All Tulsi Clusters (4 Groups)</option>
            <option value="Cluster 1">Cluster 1: Healthy Control (Pots 1–4)</option>
            <option value="Cluster 2">Cluster 2: Water Stress (Pots 5–8)</option>
            <option value="Cluster 3">Cluster 3: Light Stress (Pots 9–12)</option>
            <option value="Cluster 4">Cluster 4: Nutrient Stress (Pots 13–16)</option>
          </select>

          {/* Node filter */}
          <select
            value={selectedNode}
            onChange={(e) => setSelectedNode(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-2 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All ESP32 Nodes</option>
            <option value="1">Node 1 (Multiplexer Points 1–12)</option>
            <option value="2">Node 2 (Direct Analog Points 1–4)</option>
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

      {/* Grid Mode */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {filteredPlants.map((plant) => {
            const reading = plant.latestReading;
            const pred = plant.conditionPrediction;

            return (
              <div
                key={plant.id}
                onClick={() => onNavigate('plant-detail', plant.id)}
                className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:shadow-card-hover hover:border-[#2E8B57] transition-all cursor-pointer flex flex-col justify-between group"
              >
                <div>
                  {/* Card Header with Pot Badge and Stress Condition */}
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="w-8 h-8 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-extrabold text-xs shadow-sm">
                        #{plant.id}
                      </span>
                      <div>
                        <span className="text-[10px] font-bold uppercase tracking-wider text-[#6B786E]">
                          {plant.cluster}
                        </span>
                        <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors leading-tight">
                          {plant.name}
                        </h3>
                      </div>
                    </div>
                    <StatusBadge status={plant.stressType} size="sm" />
                  </div>

                  <p className="text-[11px] text-stone-500 italic mb-3">
                    {plant.variety} • {plant.species}
                  </p>

                  {/* Node & Point mapping */}
                  <div className="flex items-center justify-between text-[11px] text-stone-600 bg-stone-50 px-2.5 py-1.5 rounded-lg mb-3 border border-stone-200/60 font-mono">
                    <span>Node {plant.nodeId}</span>
                    <span>Point {plant.samplingPoint}</span>
                  </div>

                  {/* Sensor metrics snapshot */}
                  {reading ? (
                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                      <div className="p-2 rounded-lg bg-[#EDF4EB]/70 border border-[#DDE8D8]">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Droplets className="w-3 h-3 text-[#2E8B57]" /> Soil
                        </div>
                        <span className="font-bold text-[#1C2B23]">{reading.soil.toFixed(1)}%</span>
                      </div>
                      <div className="p-2 rounded-lg bg-stone-50 border border-stone-200/60">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Thermometer className="w-3 h-3 text-amber-600" /> Temp
                        </div>
                        <span className="font-bold text-[#1C2B23]">{reading.temperature.toFixed(1)}°C</span>
                      </div>
                      <div className="p-2 rounded-lg bg-stone-50 border border-stone-200/60">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Wind className="w-3 h-3 text-sky-600" /> Humidity
                        </div>
                        <span className="font-bold text-[#1C2B23]">{reading.humidity.toFixed(1)}%</span>
                      </div>
                      <div className="p-2 rounded-lg bg-stone-50 border border-stone-200/60">
                        <div className="flex items-center gap-1 text-[11px] text-stone-600 mb-0.5">
                          <Sun className="w-3 h-3 text-yellow-600" /> Light
                        </div>
                        <span className="font-bold text-[#1C2B23]">{Math.round(reading.light).toLocaleString()} lx</span>
                      </div>
                    </div>
                  ) : null}

                  {/* Simple Care Tip Box */}
                  <div className="p-2 rounded-lg bg-stone-50 border border-stone-200 text-[11px] text-stone-700 mb-3">
                    {plant.simpleCareTip}
                  </div>
                </div>

                {/* Footer */}
                <div className="pt-3 border-t border-[#DCE4D9] flex items-center justify-between text-[11px] text-stone-500">
                  <span>{reading ? formatRelativeTime(reading.timestamp) : 'Ready'}</span>
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
                  <th className="py-3 px-4 font-semibold">Cluster</th>
                  <th className="py-3 px-4 font-semibold">ESP32 Node / Channel</th>
                  <th className="py-3 px-4 font-semibold">Soil (%)</th>
                  <th className="py-3 px-4 font-semibold">Temp (°C)</th>
                  <th className="py-3 px-4 font-semibold">Humidity (%)</th>
                  <th className="py-3 px-4 font-semibold">Light (lux)</th>
                  <th className="py-3 px-4 font-semibold">Diagnosis</th>
                  <th className="py-3 px-4 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DCE4D9]/60">
                {filteredPlants.map((plant) => {
                  const r = plant.latestReading;
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
                        Node {plant.nodeId} • Point {plant.samplingPoint}
                      </td>
                      <td className="py-3 px-4 font-bold text-[#2E8B57]">
                        {r ? `${r.soil.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-3 px-4 font-medium text-[#1C2B23]">
                        {r ? `${r.temperature.toFixed(1)}°C` : 'N/A'}
                      </td>
                      <td className="py-3 px-4 font-medium text-[#1C2B23]">
                        {r ? `${r.humidity.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-3 px-4 font-medium text-[#1C2B23]">
                        {r ? `${Math.round(r.light).toLocaleString()} lx` : 'N/A'}
                      </td>
                      <td className="py-3 px-4">
                        <StatusBadge status={plant.stressType} size="sm" />
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onNavigate('plant-detail', plant.id);
                          }}
                          className="px-2.5 py-1 text-xs rounded bg-[#EDF4EB] text-[#2E8B57] font-semibold hover:bg-[#DDE8D8] transition-colors"
                        >
                          Dossier
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
