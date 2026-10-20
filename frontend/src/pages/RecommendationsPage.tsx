import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Filter,
  Search,
  CheckCircle2,
  Clock,
  AlertTriangle,
  BookOpen,
  Info,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  Recommendation,
  RecommendationPriority,
  RecommendationStatus,
} from '../types';
import { RecommendationCard } from '../components/shared/RecommendationCard';
import { EmptyState } from '../components/shared/EmptyState';
import { NavRoute } from '../components/layout/Sidebar';

interface RecommendationsPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const RecommendationsPage: React.FC<RecommendationsPageProps> = ({ onNavigate }) => {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [selectedPriority, setSelectedPriority] = useState<string>('all');
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    setLoading(true);
    try {
      const recs = await apiClient.getRecommendations();
      setRecommendations(recs);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateStatus = async (id: string, status: RecommendationStatus) => {
    const updated = await apiClient.updateRecommendationStatus(id, status);
    if (updated) {
      setRecommendations((prev) => prev.map((r) => (r.id === id ? updated : r)));
    }
  };

  const filteredRecs = recommendations.filter((r) => {
    if (selectedStatus !== 'all' && r.status !== selectedStatus) return false;
    if (selectedPriority !== 'all' && r.priority !== selectedPriority) return false;
    if (selectedCluster !== 'all' && r.cluster !== selectedCluster) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchTitle = r.title.toLowerCase().includes(q);
      const matchPlant = String(r.plant_id).includes(q) || r.plant_name.toLowerCase().includes(q);
      const matchReason = r.reason.toLowerCase().includes(q);
      return matchTitle || matchPlant || matchReason;
    }
    return true;
  });

  // Counts
  const newCount = recommendations.filter((r) => r.status === 'new').length;
  const inProgressCount = recommendations.filter((r) => r.status === 'in_progress').length;
  const completedCount = recommendations.filter((r) => r.status === 'completed').length;

  return (
    <div className="space-y-6">
      {/* Header & Status Summary */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              Agronomic Decision Support
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              Heuristic Rule Engine v2.1
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            Evidence-Based Plant Care Recommendations
          </h1>
          <p className="text-xs text-[#6B786E] mt-1 max-w-2xl leading-relaxed">
            Directives derived from calibrated sensor trends, persistent threshold exceedances, and species agronomic tolerances. Every action is traceable to physical readings.
          </p>
        </div>

        {/* Status Counters */}
        <div className="flex items-center gap-2 text-xs">
          <div className="px-3 py-2 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 font-semibold text-center min-w-[70px]">
            <span className="block text-sm font-bold">{newCount}</span>
            <span className="text-[10px] text-amber-700">New Actions</span>
          </div>
          <div className="px-3 py-2 rounded-xl bg-blue-50 border border-blue-200 text-blue-900 font-semibold text-center min-w-[70px]">
            <span className="block text-sm font-bold">{inProgressCount}</span>
            <span className="text-[10px] text-blue-700">In Progress</span>
          </div>
          <div className="px-3 py-2 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8] text-[#173F32] font-semibold text-center min-w-[70px]">
            <span className="block text-sm font-bold text-[#2E8B57]">{completedCount}</span>
            <span className="text-[10px] text-stone-600">Resolved</span>
          </div>
        </div>
      </div>

      {/* Advisory Guidance Notice per Prompt Rule */}
      <div className="p-4 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8] text-xs text-[#173F32] flex items-start gap-3">
        <Info className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <strong>Reviewable Action Framework:</strong> Recommendations are derived from agronomic sensor thresholds and validated guidelines. SHAP values are explanatory and not used alone to trigger automated care. A single anomalous or low moisture reading requires physical verification before irrigation dispatch.
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 flex-1">
          {/* Search */}
          <div className="relative min-w-[200px] flex-1 max-w-xs">
            <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Filter by action or plant..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full text-xs bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg pl-9 pr-3 py-1.5 outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
            />
          </div>

          {/* Status Filter */}
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Statuses</option>
            <option value="new">New</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
            <option value="dismissed">Dismissed</option>
          </select>

          {/* Priority Filter */}
          <select
            value={selectedPriority}
            onChange={(e) => setSelectedPriority(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Priorities</option>
            <option value="high">High Priority</option>
            <option value="medium">Medium Priority</option>
            <option value="low">Low Priority</option>
          </select>

          {/* Cluster Filter */}
          <select
            value={selectedCluster}
            onChange={(e) => setSelectedCluster(e.target.value)}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Tulsi Clusters</option>
            <option value="Cluster 1">Cluster 1 (Healthy Control)</option>
            <option value="Cluster 2">Cluster 2 (Water Stress)</option>
            <option value="Cluster 3">Cluster 3 (Light Stress)</option>
            <option value="Cluster 4">Cluster 4 (Nutrient Stress)</option>
          </select>
        </div>

        <span className="text-xs font-mono text-stone-500">
          Showing {filteredRecs.length} of {recommendations.length} directives
        </span>
      </div>

      {/* Recommendations Cards Grid */}
      {filteredRecs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredRecs.map((rec) => (
            <RecommendationCard
              key={rec.id}
              recommendation={rec}
              onUpdateStatus={handleUpdateStatus}
              onViewPlant={(plantId) => onNavigate('plant-detail', plantId)}
            />
          ))}
        </div>
      ) : (
        <EmptyState
          title="No Recommendations Match Active Filter"
          description="Try broadening your filter criteria or changing the status selector."
        />
      )}
    </div>
  );
};
