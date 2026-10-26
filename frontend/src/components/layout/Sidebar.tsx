import React from 'react';
import {
  LayoutDashboard,
  Activity,
  Trees,
  BrainCircuit,
  PieChart,
  Target,
  Sparkles,
  ScanSearch,
  FileSpreadsheet,
  Server,
  History,
  Settings,
  Leaf,
  ChevronRight,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { useHealth } from '../../hooks/useHealth';

export type NavRoute =
  | 'overview'
  | 'live-monitoring'
  | 'plants'
  | 'plant-detail'
  | 'ai-insights'
  | 'anomalies'
  | 'shap-analysis'
  | 'model-performance'
  | 'recommendations'
  | 'data-logs'
  | 'devices-network'
  | 'sessions'
  | 'settings';

interface SidebarProps {
  currentRoute: NavRoute;
  onNavigate: (route: NavRoute) => void;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentRoute,
  onNavigate,
  onCloseMobile,
}) => {
  const { health, error: healthError, mlConnected } = useHealth();
  const aiState = healthError ? 'backend' : health === null ? 'checking' : mlConnected ? 'connected' : 'unreachable';
  const aiOnline = aiState === 'connected';

  const navSections = [
    {
      group: 'Workspace',
      items: [
        { id: 'overview', label: 'Overview', icon: LayoutDashboard },
        { id: 'live-monitoring', label: 'Live Monitoring', icon: Activity },
        { id: 'plants', label: 'Plants', icon: Trees },
      ],
    },
    {
      group: 'Intelligence',
      items: [
        { id: 'ai-insights', label: 'AI Insights', icon: BrainCircuit },
        { id: 'anomalies', label: 'Anomalies', icon: ScanSearch },
        { id: 'shap-analysis', label: 'SHAP Analysis', icon: PieChart },
        { id: 'model-performance', label: 'Model Performance', icon: Target },
        { id: 'recommendations', label: 'Recommendations', icon: Sparkles },
      ],
    },
    {
      group: 'Operations',
      items: [
        { id: 'data-logs', label: 'Data Logs', icon: FileSpreadsheet },
        { id: 'devices-network', label: 'Devices & Network', icon: Server },
        { id: 'sessions', label: 'Sessions', icon: History },
      ],
    },
    {
      group: 'System',
      items: [
        { id: 'settings', label: 'Settings', icon: Settings },
      ],
    },
  ];

  const handleItemClick = (id: string) => {
    onNavigate(id as NavRoute);
    if (onCloseMobile) onCloseMobile();
  };

  return (
    <aside className="w-64 bg-[#102F26] text-[#DDE8D8] flex flex-col h-screen select-none border-r border-[#173F32]/60 shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-[#173F32] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#2E8B57] text-white flex items-center justify-center shadow-md shadow-black/20">
            <Leaf className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight leading-tight">
              Plant Monitor
            </h1>
            <div className="flex items-center gap-1.5 text-[11px] text-[#79B86A] font-medium">
              <Leaf className="w-3 h-3" />
              <span>Tulsi stress monitoring</span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
        {navSections.map((sec, idx) => (
          <div key={idx}>
            <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-[#79B86A]/70">
              {sec.group}
            </div>
            <nav className="space-y-1">
              {sec.items.map((item) => {
                const Icon = item.icon;
                const isActive =
                  currentRoute === item.id ||
                  (item.id === 'plants' && currentRoute === 'plant-detail');

                return (
                  <button
                    key={item.id}
                    onClick={() => handleItemClick(item.id)}
                    className={cn(
                      'w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all group',
                      isActive
                        ? 'bg-[#173F32] text-white shadow-sm border border-[#2E8B57]/40 font-semibold'
                        : 'text-[#DDE8D8]/80 hover:text-white hover:bg-[#173F32]/50'
                    )}
                  >
                    <div className="flex items-center gap-2.5">
                      <Icon
                        className={cn(
                          'w-4 h-4 transition-colors',
                          isActive ? 'text-[#79B86A]' : 'text-[#79B86A]/70 group-hover:text-white'
                        )}
                      />
                      <span>{item.label}</span>
                    </div>
                    {isActive && (
                      <ChevronRight className="w-3.5 h-3.5 text-[#79B86A]" />
                    )}
                  </button>
                );
              })}
            </nav>
          </div>
        ))}
      </div>

      {/* AI model connection indicator (from /api/health) */}
      <div className="p-4 border-t border-[#173F32] bg-[#0c241d] text-xs" data-testid="sidebar-model-status">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 relative">
              <span
                className={cn(
                  'absolute inline-flex h-full w-full rounded-full opacity-75',
                  aiOnline ? 'animate-ping bg-emerald-400' : 'bg-red-400'
                )}
              />
              <span
                className={cn(
                  'relative inline-flex rounded-full h-2 w-2',
                  aiOnline ? 'bg-[#2E8B57]' : aiState === 'checking' ? 'bg-stone-400' : 'bg-red-500'
                )}
              />
            </span>
            <span className="font-semibold text-white text-[11px]">
              {aiState === 'connected'
                ? 'AI model connected'
                : aiState === 'unreachable'
                  ? 'AI model unreachable'
                  : aiState === 'backend'
                    ? 'Backend unreachable'
                    : 'Checking AI model...'}
            </span>
          </div>
        </div>

        <div className="text-[10px] text-stone-400 space-y-0.5 font-mono">
          <div className="flex justify-between">
            <span>Model:</span>
            <span className="text-stone-300">{health?.modelVersion ? `v${health.modelVersion}` : 'n/a'}</span>
          </div>
          <div className="flex justify-between">
            <span>Dataset:</span>
            <span className="text-stone-300">{health?.datasetName ?? 'n/a'}</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
