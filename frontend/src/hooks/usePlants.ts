import { useContext } from 'react';
import { apiClient } from '../api/client';
import type { Plant, PlantMonitoring } from '../types';
import { useAsync, type AsyncState } from './useApi';
import { SharedDataContext } from './sharedData';

export interface PlantsState extends AsyncState<Plant[]> {
  total: number;
}

function toPlantsState(state: AsyncState<{ items: Plant[]; total: number }>): PlantsState {
  const page = state.data;
  return { ...state, data: page ? page.items : null, total: page ? page.total : 0 };
}

/** Fetches all plants itself. The app shell uses this once and shares the result via `SharedDataContext`. */
export function usePlantsSource(): PlantsState {
  return toPlantsState(useAsync(() => apiClient.getPlantsPage(), ['plants']));
}

/** All plants with latest reading and latest ML prediction (shared with the app shell when available). */
export function usePlants(): PlantsState {
  const shared = useContext(SharedDataContext).plants;
  const own = toPlantsState(useAsync(() => apiClient.getPlantsPage(), ['plants'], { enabled: shared === null }));
  return shared ?? own;
}

/** Plant + recent series + prediction history + active recommendations. */
export function usePlantMonitoring(id: number | string | null | undefined, limit = 84): AsyncState<PlantMonitoring | null> {
  return useAsync(() => apiClient.getPlantMonitoring(id as number | string, limit), ['plantMonitoring', id, limit], {
    enabled: id !== null && id !== undefined,
  });
}
