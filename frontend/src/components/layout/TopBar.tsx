import React, { useState } from 'react';
import {
  Menu,
  Search,
  Bell,
  RefreshCw,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  X,
} from 'lucide-react';
import { NavRoute } from './Sidebar';

interface TopBarProps {
  currentRoute: NavRoute;
  activePlantId?: number | string | null;
  onOpenMobile: () => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
  notificationCount?: number;
  onNavigate: (route: NavRoute) => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  currentRoute,
  activePlantId,
  onOpenMobile,
  onRefresh,
  isRefreshing = false,
  notificationCount = 2,
  onNavigate,
}) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Page titles mapping
  const routeTitles: Record<NavRoute, { title: string; category: string }> = {
    overview: { title: 'Tulsi Telemetry & Collection Health', category: 'Workspace' },
    'live-monitoring': { title: 'Live Sensor Stream Monitoring', category: 'Workspace' },
    plants: { title: 'Tulsi Stress Trial Specimens (16 Plants)', category: 'Workspace' },
    'plant-detail': {
      title: activePlantId ? `Specimen Dossier — ${String(activePlantId).toUpperCase()}` : 'Plant Specification',
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
      <div className="flex items-center gap-3">
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
      <div className="flex items-center gap-3">
        {/* Contextual Search */}
        <div className="relative hidden lg:block w-64">
          <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search plant, node, record..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full text-xs bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg pl-9 pr-3 py-1.5 outline-none focus:border-[#2E8B57] focus:bg-white transition-all text-[#1C2B23]"
          />
        </div>

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
            title="System Notifications"
          >
            <Bell className="w-4 h-4" />
            {notificationCount > 0 && (
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-[#C98227]" />
            )}
          </button>

          {/* Notifications Dropdown */}
          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-dropdown border border-[#DCE4D9] py-3 z-50 text-xs animate-fadeIn">
              <div className="px-4 pb-2 border-b border-[#DCE4D9] flex items-center justify-between">
                <span className="font-bold text-[#1C2B23]">Active Telemetry Alerts</span>
                <button
                  onClick={() => setShowNotifications(false)}
                  className="text-stone-400 hover:text-stone-600"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="divide-y divide-[#DCE4D9]/60 max-h-64 overflow-y-auto">
                <div
                  onClick={() => {
                    onNavigate('recommendations');
                    setShowNotifications(false);
                  }}
                  className="p-3 hover:bg-[#EDF4EB]/50 cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-1.5 text-amber-700 font-semibold mb-1">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                    <span>Tulsi #5 (Cluster 2): Water Stress Deficit</span>
                  </div>
                  <p className="text-[11px] text-[#6B786E] leading-normal">
                    Soil moisture registered 16.4% in drought trial. Regulated hydration recommendation created.
                  </p>
                </div>

                <div
                  onClick={() => {
                    onNavigate('recommendations');
                    setShowNotifications(false);
                  }}
                  className="p-3 hover:bg-[#EDF4EB]/50 cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-1.5 text-yellow-700 font-semibold mb-1">
                    <AlertTriangle className="w-3.5 h-3.5 text-yellow-600" />
                    <span>Tulsi #9 (Cluster 3): Low PAR Light Stress</span>
                  </div>
                  <p className="text-[11px] text-[#6B786E] leading-normal">
                    Illumination measured at 450 lux. Supplemental LED photoperiod action suggested.
                  </p>
                </div>
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
          <span>UTC 12:00</span>
        </div>
      </div>
    </header>
  );
};
