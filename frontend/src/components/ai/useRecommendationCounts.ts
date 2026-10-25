import { apiClient } from '../../api/client';
import { useAsync, type AsyncState } from '../../hooks/useApi';
import type { RecommendationStatus } from '../../types';

export const RECOMMENDATION_STATUSES: RecommendationStatus[] = ['new', 'in_progress', 'completed', 'dismissed', 'superseded'];

export type RecommendationCounts = Record<RecommendationStatus, number>;

/**
 * Counts by status from the server (one `limit=1` request per status, using the X-Total-Count header),
 * so the totals do not depend on the page that is displayed. Refetches with the global data version.
 */
export function useRecommendationCounts(): AsyncState<RecommendationCounts> {
  return useAsync(async () => {
    const pages = await Promise.all(
      RECOMMENDATION_STATUSES.map((status) => apiClient.getRecommendationsPage({ status, limit: 1, includeSuperseded: status === 'superseded' })),
    );
    const counts = {} as RecommendationCounts;
    RECOMMENDATION_STATUSES.forEach((status, i) => {
      counts[status] = pages[i].total;
    });
    return counts;
  }, ['recommendationCounts']);
}
