import { apiClient, type RecommendationFilter } from '../api/client';
import type { Recommendation } from '../types';
import { useAsync, type AsyncState } from './useApi';

export interface RecommendationsState extends AsyncState<Recommendation[]> {
  total: number;
}

/** A page of recommendations. Status changes go through `useRecommendationStatus`. */
export function useRecommendations(filter?: RecommendationFilter): RecommendationsState {
  const state = useAsync(() => apiClient.getRecommendationsPage(filter), ['recommendations', filter]);
  const page = state.data;
  return { ...state, data: page ? page.items : null, total: page ? page.total : 0 };
}
