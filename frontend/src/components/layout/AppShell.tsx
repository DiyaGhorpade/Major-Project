import React, { useState } from 'react';
import { Sidebar, NavRoute } from './Sidebar';
import { TopBar } from './TopBar';
import { DemoDataBanner } from '../shared/DemoDataBanner';
import { apiClient } from '../../api/client';
import { X } from 'lucide-react';

interface AppShellProps {
  currentRoute: NavRoute;
  activePlantId?: number | string | null;
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentRoute,
  activePlantId,
  onNavigate,
  children,
}) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isDemo, setIsDemo] = useState(apiClient.isDemoMode());
  const [isModelConnected, setIsModelConnected] = useState(apiClient.isModelConnected());
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleToggleDemo = (demo: boolean) => {
    apiClient.setDemoMode(demo);
    setIsDemo(demo);
  };

  const handleToggleModel = (connected: boolean) => {
    apiClient.setModelConnected(connected);
    setIsModelConnected(connected);
  };

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setIsRefreshing(false);
    }, 600);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#F6F7F0] text-[#1C2B23]">
      {/* Desktop Persistent Sidebar */}
      <div className="hidden md:flex shrink-0">
        <Sidebar
          currentRoute={currentRoute}
          onNavigate={(route) => onNavigate(route)}
          isJetsonOnline={true}
          isDemo={isDemo}
        />
      </div>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden bg-black/50 backdrop-blur-sm animate-fadeIn">
          <div className="relative flex-1 max-w-xs w-full bg-[#102F26]">
            <Sidebar
              currentRoute={currentRoute}
              onNavigate={(route) => onNavigate(route)}
              isJetsonOnline={true}
              isDemo={isDemo}
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
          isRefreshing={isRefreshing}
          notificationCount={2}
          onNavigate={(route) => onNavigate(route)}
        />

        {/* Demo Mode / Model Disconnected Banner */}
        <DemoDataBanner
          isDemo={isDemo}
          onToggleDemo={handleToggleDemo}
          isModelConnected={isModelConnected}
          onToggleModel={handleToggleModel}
        />

        {/* Page Canvas */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
};
