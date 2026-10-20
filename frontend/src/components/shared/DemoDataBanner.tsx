import React from 'react';
import { Activity, ShieldAlert, Cpu, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';

interface DemoDataBannerProps {
  isDemo: boolean;
  onToggleDemo: (demo: boolean) => void;
  isModelConnected?: boolean;
  onToggleModel?: (connected: boolean) => void;
}

export const DemoDataBanner: React.FC<DemoDataBannerProps> = ({
  isDemo,
  onToggleDemo,
  isModelConnected = true,
  onToggleModel,
}) => {
  if (!isDemo && isModelConnected) {
    return null;
  }

  return (
    <div className="bg-amber-50/90 border-b border-amber-200/80 px-4 py-2.5 text-xs text-amber-900 flex flex-wrap items-center justify-between gap-3 shadow-sm">
      <div className="flex items-center gap-2">
        <span className="flex h-2 w-2 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
        </span>
        <span className="font-semibold uppercase tracking-wider text-[11px] bg-amber-200/80 text-amber-900 px-2 py-0.5 rounded">
          {isDemo ? 'Demo Mode Active' : 'Live Mode'}
        </span>
        <span className="text-amber-800">
          {isDemo
            ? 'Displaying synthetic sensor telemetry & baseline ML predictions. Values do not represent physical hardware.'
            : !isModelConnected
            ? 'Connected to live Jetson ingest stream. Edge ML model is currently detached / offline.'
            : ''}
        </span>
      </div>

      <div className="flex items-center gap-3">
        {onToggleModel && (
          <button
            type="button"
            onClick={() => onToggleModel(!isModelConnected)}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-100 hover:bg-amber-200 text-amber-900 font-medium transition-colors border border-amber-300"
            title="Simulate edge inference service online vs offline"
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Model: {isModelConnected ? 'Connected' : 'Offline'}</span>
          </button>
        )}

        <button
          type="button"
          onClick={() => onToggleDemo(!isDemo)}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-900 text-white font-medium hover:bg-amber-800 transition-colors shadow-sm"
        >
          <Activity className="w-3.5 h-3.5" />
          <span>{isDemo ? 'Switch to Live API' : 'Return to Demo Mode'}</span>
        </button>
      </div>
    </div>
  );
};
