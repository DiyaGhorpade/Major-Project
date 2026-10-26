import React, { useEffect, useState } from 'react';
import { Menu, Search, Bell, RefreshCw, AlertTriangle, Play, RotateCcw, Loader2, X } from 'lucide-react';
import { NavRoute } from './Sidebar';
import { usePipeline } from '../../hooks/usePipeline';
import { formatCount, formatIstClock, formatRelativeTime } from '../../lib/utils';
import type { Alert } from '../../types';

interface TopBarProps {
  currentRoute: NavRoute;
  activePlantId?: number | string | null;
  onOpenMobile: () => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
  /** Recent alerts from /api/monitoring/overview (empty = no badge). */
  alerts?: Alert[];
  /** Plants from the data, used by the search box and the page titles (empty until loaded). */
  plants?: Array<{ id: number; name: string }>;
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  currentRoute,
  activePlantId,
  onOpenMobile,
  onRefresh,
  isRefreshing = false,
  alerts = [],
  plants = [],
  onNavigate,
}) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMiss, setSearchMiss] = useState(false);
  const [clock, setClock] = useState(() => formatIstClock());
  const { run, isActive, hasCompletedRun, processed, total, percent } = usePipeline();

  useEffect(() => {
    const id = setInterval(() => setClock(formatIstClock()), 15000);
    return () => clearInterval(id);
  }, []);

  const plantCount = plants.length;
  const plantCountText = plantCount > 0 ? ` (${plantCount} Plants)` : '';

  // Jump to a plant by id ("5", "#5", "pot 5") or by (part of) its name.
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const q = searchQuery.trim().toLowerCase();
    if (!q) return;
    const idMatch = /^(?:pot\s*)?#?(\d+)$/.exec(q);
    const found = idMatch
      ? plants.find((p) => p.id === Number(idMatch[1]))
      : plants.find((p) => p.name.toLowerCase().includes(q));
    if (found) {
      setSearchMiss(false);
      setSearchQuery('');
      onNavigate('plant-detail', found.id);
    } else {
      setSearchMiss(true);
    }
  };

  const runLabel = hasCompletedRun ? 'Re-run' : 'Run';
  const RunIcon = hasCompletedRun ? RotateCcw : Play;
  const starting = total === 0;
  const progressText = starting ? 'Starting analysis...' : `Analyzing ${formatCount(processed)} / ${formatCount(total)} readings...`;
  const compactText = starting ? 'Starting...' : `${Math.round(percent)}%`;

  // Page titles mapping
  const routeTitles: Record<NavRoute, { title: string; category: string }> = {
    overview: { title: 'Tulsi Telemetry & Collection Health', category: 'Workspace' },
    'live-monitoring': { title: `Plant Sensor Monitoring${plantCountText}`, category: 'Workspace' },
    plants: { title: `Tulsi Stress Trial Specimens${plantCountText}`, category: 'Workspace' },
    'plant-detail': {
      title: activePlantId ? `Specimen Dossier: Pot ${String(activePlantId)}` : 'Plant Specification',
      category: 'Workspace / Plants',
    },
    'ai-insights': { title: 'Tulsi Stress ML Inference Insights', category: 'Intelligence' },
    'shap-analysis': { title: 'SHAP Explainability Workspace', category: 'Intelligence' },
    'model-performance': { title: 'Model Evaluation & Benchmark Metrics', category: 'Intelligence' },
    recommendations: { title: 'Agronomic Care Recommendations', category: 'Intelligence' },
    'data-logs': { title: 'Telemetry Data Logs & CSV Export', category: 'Operations' },
    'devices-network': { title: 'Devices & Network Infrastructure', category: 'Operations' },
    sessions: { title: 'Ingestion Sessions History', category: 'Operations' },
    settings: { title: 'System Configuration & Calibration', category: 'System' },
  };

  const current = routeTitles[currentRoute] || { title: 'Plant Monitor', category: 'Dashboard' };

  return (
    <header className="h-16 bg-white border-b border-[#DCE4D9] px-4 md:px-6 flex items-center justify-between sticky top-0 z-30 shadow-sm">
      {/* Left: Mobile hamburger & Page Title */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onOpenMobile}
          className="md:hidden p-2 rounded-lg text-stone-600 hover:bg-stone-100 transition-colors"
          aria-label="Toggle menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          <div className="text-[10px] uppercase font-bold tracking-wider text-[#426B4F]">
            {current.category}
          </div>
          <h2 className="text-sm md:text-base font-bold text-[#1C2B23] tracking-tight">
            {current.title}
          </h2>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-3 shrink-0">
        {/* Run pipeline + progress */}
        <div className="flex items-center gap-3">
          {isActive && (
            <div className="w-20 sm:w-48" data-testid="run-progress">
              <div className="text-[11px] text-[#426B4F] font-medium leading-tight truncate" aria-live="polite">
                <span className="sm:hidden">{compactText}</span>
                <span className="hidden sm:inline">{progressText}</span>
              </div>
              <div
                className="mt-1 h-1.5 rounded-full bg-[#DCE4D9] overflow-hidden"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(percent)}
              >
                <div className="h-full bg-[#2E8B57] transition-all duration-500" style={{ width: `${Math.min(100, percent)}%` }} />
              </div>
            </div>
          )}
          <button
            type="button"
            onClick={() => void run()}
            disabled={isActive}
            title={isActive ? 'Analysis in progress' : 'Analyze the plant data with the ML model'}
            className="inline-flex items-center gap-1.5 shrink-0 whitespace-nowrap px-3 sm:px-4 py-2 rounded-lg bg-[#2E8B57] hover:bg-[#236e44] disabled:opacity-60 disabled:cursor-not-allowed text-white text-sm font-semibold shadow-sm transition-colors"
          >
            {isActive ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <RunIcon className="w-4 h-4" aria-hidden="true" />}
            <span>{runLabel}</span>
          </button>
        </div>

        {/* Jump to a plant */}
        <form onSubmit={handleSearch} role="search" className="relative hidden lg:block w-64">
          <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" aria-hidden="true" />
          <input
            type="text"
            list="topbar-plants"
            aria-label="Jump to a plant by number or name"
            placeholder="Jump to plant (number or name)..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setSearchMiss(false);
            }}
            className={`w-full text-xs bg-[#F6F7F0] border rounded-lg pl-9 pr-3 py-1.5 outline-none focus:border-[#2E8B57] focus:bg-white transition-all text-[#1C2B23] ${searchMiss ? 'border-red-400' : 'border-[#DCE4D9]'}`}
          />
          <datalist id="topbar-plants">
            {plants.map((p) => (
              <option key={p.id} value={p.name} />
            ))}
          </datalist>
          {searchMiss && (
            <span role="alert" className="absolute left-0 top-full mt-1 text-[11px] text-red-700">
              No matching plant.
            </span>
          )}
        </form>

        {/* Refresh button */}
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="p-2 rounded-lg text-stone-600 hover:text-stone-900 hover:bg-[#EDF4EB] transition-colors border border-transparent hover:border-[#DDE8D8]"
          title="Refresh Telemetry"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-[#2E8B57]' : ''}`} />
        </button>

        {/* Notification Indicator only when meaningful data exists */}
        <div className="relative">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className="p-2 rounded-lg text-stone-600 hover:text-stone-900 hover:bg-[#EDF4EB] transition-colors relative border border-transparent hover:border-[#DDE8D8]"
            title="Recent alerts"
          >
            <Bell className="w-4 h-4" />
            {alerts.length > 0 && (
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-[#C98227]" />
            )}
          </button>

          {/* Notifications Dropdown */}
          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-dropdown border border-[#DCE4D9] py-3 z-50 text-xs animate-fadeIn">
              <div className="px-4 pb-2 border-b border-[#DCE4D9] flex items-center justify-between">
                <span className="font-bold text-[#1C2B23]">Recent Alerts</span>
                <button
                  onClick={() => setShowNotifications(false)}
                  className="text-stone-400 hover:text-stone-600"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="divide-y divide-[#DCE4D9]/60 max-h-64 overflow-y-auto">
                {alerts.length === 0 && (
                  <div className="p-4 text-center text-[11px] text-[#6B786E]">No alerts. Press Run to analyze the dataset.</div>
                )}
                {alerts.slice(0, 8).map((alert, idx) => (
                  <div
                    key={`${alert.kind}-${alert.recommendationId ?? alert.recordId ?? idx}`}
                    onClick={() => {
                      if (alert.kind === 'recommendation') onNavigate('recommendations');
                      else onNavigate('plant-detail', alert.plantId);
                      setShowNotifications(false);
                    }}
                    className="p-3 hover:bg-[#EDF4EB]/50 cursor-pointer transition-colors"
                  >
                    <div className="flex items-center gap-1.5 text-amber-700 font-semibold mb-1">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                      <span className="truncate">
                        {alert.plantName}
                        {alert.kind === 'anomaly' ? ' - unusual reading' : ''}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#6B786E] leading-normal line-clamp-2">{alert.message}</p>
                    <p className="text-[10px] text-stone-400 mt-0.5">{formatRelativeTime(alert.timestamp)}</p>
                  </div>
                ))}
              </div>

              <div className="px-4 pt-2 border-t border-[#DCE4D9] text-center">
                <button
                  onClick={() => {
                    onNavigate('recommendations');
                    setShowNotifications(false);
                  }}
                  className="text-[11px] font-bold text-[#2E8B57] hover:underline"
                >
                  View All Recommendations →
                </button>
              </div>
            </div>
          )}
        </div>

        {/* System Time badge */}
        <div className="hidden sm:flex items-center gap-2 pl-2 border-l border-[#DCE4D9] text-xs text-[#6B786E] font-mono">
          <span className="w-2 h-2 rounded-full bg-[#2E8B57]" />
          <span>IST {clock}</span>
        </div>
      </div>
    </header>
  );
};
