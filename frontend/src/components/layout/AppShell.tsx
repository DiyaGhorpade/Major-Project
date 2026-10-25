import React, { useCallback, useContext, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { Sidebar, NavRoute } from './Sidebar';
import { TopBar } from './TopBar';
import { SyntheticBanner } from '../shared/SyntheticBanner';
import { ErrorBanner } from '../shared/ErrorBanner';
import { DataVersionContext } from '../../hooks/dataVersion';
import { PipelineContext, usePipeline, usePipelineMachine } from '../../hooks/usePipeline';
import { HealthContext, useHealth, useHealthPolling } from '../../hooks/useHealth';
import { useOverviewSource } from '../../hooks/useData';
import { usePlantsSource } from '../../hooks/usePlants';
import { SharedDataContext } from '../../hooks/sharedData';

interface AppShellProps {
  currentRoute: NavRoute;
  activePlantId?: number | string | null;
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
  children: React.ReactNode;
}

/** Provides the data-version, pipeline and health contexts, then renders the shell. */
export const AppShell: React.FC<AppShellProps> = (props) => {
  const [dataVersion, setDataVersion] = useState(0);
  const bumpDataVersion = useCallback(() => setDataVersion((v) => v + 1), []);
  const versionValue = useMemo(() => ({ dataVersion, bumpDataVersion }), [dataVersion, bumpDataVersion]);

  return (
    <DataVersionContext.Provider value={versionValue}>
      <ShellProviders bumpDataVersion={bumpDataVersion}>
        <ShellLayout {...props} />
      </ShellProviders>
    </DataVersionContext.Provider>
  );
};

const ShellProviders: React.FC<{ bumpDataVersion: () => void; children: React.ReactNode }> = ({ bumpDataVersion, children }) => {
  const pipeline = usePipelineMachine(bumpDataVersion);
  const health = useHealthPolling();
  return (
    <PipelineContext.Provider value={pipeline}>
      <HealthContext.Provider value={health}>
        <SharedDataProvider>{children}</SharedDataProvider>
      </HealthContext.Provider>
    </PipelineContext.Provider>
  );
};

/** Loads the overview and plants once for the whole app (bell, search box, pages). */
const SharedDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const overview = useOverviewSource();
  const plants = usePlantsSource();
  const value = useMemo(() => ({ overview, plants }), [overview, plants]);
  return <SharedDataContext.Provider value={value}>{children}</SharedDataContext.Provider>;
};

const ShellLayout: React.FC<AppShellProps> = ({ currentRoute, activePlantId, onNavigate, children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pipeline = usePipeline();
  const { health, error: healthError, mlUnreachable } = useHealth();
  const { overview, plants } = useContext(SharedDataContext);
  const { bumpDataVersion } = useContext(DataVersionContext);

  // Refresh button: bump the data version so every hook refetches; it spins while that refetch runs.
  const handleRefresh = () => bumpDataVersion();

  // Most specific cause first: backend down, failed run, then ML service unreachable.
  let banner: { title: string; message: string; retry?: () => void; retryLabel?: string } | null = null;
  if (healthError) {
    banner = { title: 'Backend not reachable', message: healthError.message, retry: bumpDataVersion, retryLabel: 'Retry' };
  } else if (pipeline.state === 'failed') {
    banner = {
      title: 'Analysis run failed',
      message: pipeline.message ?? 'The run failed.',
      retry: () => void pipeline.run(),
      retryLabel: 'Run again',
    };
  } else if (pipeline.error) {
    banner = { title: 'Pipeline error', message: pipeline.error.message, retry: () => void pipeline.run(), retryLabel: 'Run again' };
  } else if (mlUnreachable && health) {
    banner = {
      title: 'ML service not reachable',
      message: `ML service not reachable at ${health.mlService.url} (${health.mlService.detail}). Raw readings still work; predictions are unavailable.`,
      retry: bumpDataVersion,
      retryLabel: 'Check again',
    };
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#F6F7F0] text-[#1C2B23]">
      {/* Desktop Persistent Sidebar */}
      <div className="hidden md:flex shrink-0">
        <Sidebar currentRoute={currentRoute} onNavigate={(route) => onNavigate(route)} />
      </div>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden bg-black/50 backdrop-blur-sm animate-fadeIn">
          <div className="relative flex-1 max-w-xs w-full bg-[#102F26]">
            <Sidebar
              currentRoute={currentRoute}
              onNavigate={(route) => onNavigate(route)}
             
              onCloseMobile={() => setMobileOpen(false)}
            />
            <button
              onClick={() => setMobileOpen(false)}
              className="absolute top-4 right-4 p-2 text-stone-300 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="flex-1" onClick={() => setMobileOpen(false)} />
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        <TopBar
          currentRoute={currentRoute}
          activePlantId={activePlantId}
          onOpenMobile={() => setMobileOpen(true)}
          onRefresh={handleRefresh}
          isRefreshing={overview?.refreshing ?? false}
          alerts={overview?.data?.recentAlerts ?? []}
          plants={plants?.data ?? []}
          onNavigate={onNavigate}
        />

        <SyntheticBanner />

        {banner && (
          <ErrorBanner title={banner.title} error={banner.message} onRetry={banner.retry} retryLabel={banner.retryLabel} />
        )}

        {/* Page Canvas */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
};
