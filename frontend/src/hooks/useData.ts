import { useContext } from 'react';
import { apiClient } from '../api/client';
import type {
  AiInsights,
  Device,
  LocalShapExplanation,
  ModelMetrics,
  Overview,
  Session,
} from '../types';
import { useAsync, type AsyncState } from './useApi';
import { SharedDataContext } from './sharedData';
import type { GlobalShapResult } from '../api/client';

/** Fetches the overview itself. The app shell uses this once and shares the result via `SharedDataContext`. */
export const useOverviewSource = (): AsyncState<Overview> => useAsync(() => apiClient.getOverview(), ['overview']);

/** The shared overview from the app shell, or its own fetch when rendered without one. */
export function useOverview(): AsyncState<Overview> {
  const shared = useContext(SharedDataContext).overview;
  const own = useAsync(() => apiClient.getOverview(), ['overview'], { enabled: shared === null });
  return shared ?? own;
}

export const useInsights = (): AsyncState<AiInsights> => useAsync(() => apiClient.getInsights(), ['insights']);

export const useModelMetrics = (): AsyncState<ModelMetrics> => useAsync(() => apiClient.getModelMetrics(), ['metrics']);

export const useSessions = (): AsyncState<Session[]> => useAsync(() => apiClient.getSessions(), ['sessions']);

export const useDevices = (): AsyncState<Device[]> => useAsync(() => apiClient.getDevices(), ['devices']);

/** Null data once loaded means there is no explanation yet (press Run). */
export const useLocalShap = (plantId: number | string | null | undefined): AsyncState<LocalShapExplanation | null> =>
  useAsync(() => apiClient.getLocalShap(plantId as number | string), ['localShap', plantId], {
    enabled: plantId !== null && plantId !== undefined,
  });

/** `data.unavailable.hint` carries the command to run when the report does not exist. */
export const useGlobalShap = (): AsyncState<GlobalShapResult> => useAsync(() => apiClient.getGlobalShapResult(), ['globalShap']);

