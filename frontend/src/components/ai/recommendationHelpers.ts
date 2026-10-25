import type { ApiError } from '../../api/client';
import type { Recommendation } from '../../types';

export const PAGE_SIZE = 12;

/** Message for a failed status change; a 409 names the active recommendation that blocks reopening. */
export function describeRecommendationError(error: ApiError, known: Recommendation[]): string {
  if (error.status === 409 && error.code === 'conflict') {
    const details = error.details as { activeId?: unknown } | null;
    const activeId = details && typeof details.activeId === 'string' ? details.activeId : null;
    if (activeId) {
      const active = known.find((r) => r.id === activeId);
      const name = active ? `${activeId} ("${active.title}", ${active.plant_name})` : activeId;
      return `Cannot reopen this recommendation: another active recommendation for the same plant and category already exists (${name}). Complete or dismiss that one first.`;
    }
    return 'Cannot reopen this recommendation: another active recommendation for the same plant and category already exists. Complete or dismiss that one first.';
  }
  return error.message;
}
