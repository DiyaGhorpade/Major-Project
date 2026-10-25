import { apiClient } from '../api/client';
import type { Plant, PlantMonitoring } from '../types';
import { useAsync, type AsyncState } from './useApi';

export interface PlantsState extends AsyncState<Plant[]> {
  total: number;
}

/** All 16 plants with latest reading and latest ML prediction. */
export function usePlants(): PlantsState {
  const state = useAsync(() => apiClient.getPlantsPage(), ['plants']);
  const page = state.data;
  return { ...state, data: page ? page.items : null, total: page ? page.total : 0 };
}

/** One plant (null once loaded means the id does not exist). */
export function usePlant(id: number | string | null | undefined): AsyncState<Plant | null> {
  return useAsync(() => apiClient.getPlantById(id as number | string), ['plant', id], { enabled: id !== null && id !== undefined });
}

/** Plant + recent series + prediction history + active recommendations. */
export function usePlantMonitoring(id: number | string | null | undefined, limit = 84): AsyncState<PlantMonitoring | null> {
  return useAsync(() => apiClient.getPlantMonitoring(id as number | string, limit), ['plantMonitoring', id, limit], {
    enabled: id !== null && id !== undefined,
  });
}
