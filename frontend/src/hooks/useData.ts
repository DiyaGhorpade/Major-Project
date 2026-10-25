import { apiClient } from '../api/client';
import type {
  AiInsights,
  AnomalyRecord,
  Device,
  LocalShapExplanation,
  ModelMetrics,
  Overview,
  Prediction,
  Session,
} from '../types';
import { useAsync, type AsyncState } from './useApi';
import type { GlobalShapResult, PredictionFilter } from '../api/client';

export const useOverview = (): AsyncState<Overview> => useAsync(() => apiClient.getOverview(), ['overview']);

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

export const useAnomalies = (limit = 100): AsyncState<AnomalyRecord[]> =>
  useAsync(async () => (await apiClient.getAnomaliesPage({ limit })).items, ['anomalies', limit]);

export const usePredictions = (filter?: PredictionFilter): AsyncState<Prediction[]> =>
  useAsync(async () => (await apiClient.getPredictionsPage(filter)).items, ['predictions', filter]);
