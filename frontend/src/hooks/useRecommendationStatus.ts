import { useCallback, useRef, useState } from 'react';
import { apiClient, toApiError, type ApiError } from '../api/client';
import type { Recommendation, UserRecommendationStatus } from '../types';
import { useDataVersion } from './dataVersion';

export interface RecommendationStatusActions {
  /**
   * PATCH the status, then bump the shared data version so every list refetches. Returns the updated item, or null
   * when it failed or was skipped because a change for this id is already in flight (no duplicate PATCHes).
   * `onFailure` runs only if this is still the latest request for the id (e.g. to roll back an optimistic change).
   */
  update: (id: string, status: UserRecommendationStatus, onFailure?: () => void) => Promise<Recommendation | null>;
  /** Last failed update, e.g. a 409 `conflict` when reopening would duplicate an active recommendation. */
  error: ApiError | null;
  clearError: () => void;
  /** True while a change for this recommendation is being saved. */
  isPending: (id: string) => boolean;
}

/** The one place recommendation status changes are made (Overview, Plant detail and Recommendations pages). */
export function useRecommendationStatus(): RecommendationStatusActions {
  const { bumpDataVersion } = useDataVersion();
  const [error, setError] = useState<ApiError | null>(null);
  const [pending, setPending] = useState<ReadonlySet<string>>(() => new Set());
  const inFlight = useRef(new Set<string>());
  const latest = useRef(new Map<string, number>());
  const counter = useRef(0);

  const update = useCallback(
    async (id: string, status: UserRecommendationStatus, onFailure?: () => void) => {
      if (inFlight.current.has(id)) return null;
      const seq = ++counter.current;
      latest.current.set(id, seq);
      inFlight.current.add(id);
      setPending(new Set(inFlight.current));
      setError(null);
      try {
        const next = await apiClient.updateRecommendationStatus(id, status);
        bumpDataVersion();
        return next;
      } catch (err) {
        setError(toApiError(err));
        if (latest.current.get(id) === seq) onFailure?.();
        return null;
      } finally {
        inFlight.current.delete(id);
        setPending(new Set(inFlight.current));
      }
    },
    [bumpDataVersion],
  );

  const clearError = useCallback(() => setError(null), []);
  const isPending = useCallback((id: string) => pending.has(id), [pending]);
  return { update, error, clearError, isPending };
}
