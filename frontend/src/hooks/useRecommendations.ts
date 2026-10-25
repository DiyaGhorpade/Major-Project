import { useCallback, useState } from 'react';
import { apiClient, toApiError, type ApiError, type RecommendationFilter } from '../api/client';
import type { Recommendation, UserRecommendationStatus } from '../types';
import { useAsync, type AsyncState } from './useApi';
import { useDataVersion } from './dataVersion';

export interface RecommendationsState extends AsyncState<Recommendation[]> {
  total: number;
  /** PATCH the status, then refetch. Returns the updated item, or null on failure (see `mutationError`). */
  setStatus: (id: string, status: UserRecommendationStatus) => Promise<Recommendation | null>;
  /** Last failed update, e.g. a 409 `conflict` when reopening would duplicate an active recommendation. */
  mutationError: ApiError | null;
  clearMutationError: () => void;
}

export function useRecommendations(filter?: RecommendationFilter): RecommendationsState {
  const state = useAsync(() => apiClient.getRecommendationsPage(filter), ['recommendations', filter]);
  const { bumpDataVersion } = useDataVersion();
  const [mutationError, setMutationError] = useState<ApiError | null>(null);

  const setStatus = useCallback(
    async (id: string, status: UserRecommendationStatus) => {
      setMutationError(null);
      try {
        const updated = await apiClient.updateRecommendationStatus(id, status);
        bumpDataVersion(); // overview/top actions and every list refetch
        return updated;
      } catch (err) {
        setMutationError(toApiError(err));
        return null;
      }
    },
    [bumpDataVersion],
  );

  const page = state.data;
  return {
    ...state,
    data: page ? page.items : null,
    total: page ? page.total : 0,
    setStatus,
    mutationError,
    clearMutationError: () => setMutationError(null),
  };
}
