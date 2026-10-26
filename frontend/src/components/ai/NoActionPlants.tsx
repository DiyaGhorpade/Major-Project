import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useAsync } from '../../hooks/useApi';
import type { Plant } from '../../types';

/** Ids of plants with at least one open (new or in-progress) recommendation. Refetches with the global data version. */
function useOpenActionPlantIds() {
  return useAsync(async () => {
    const pages = await Promise.all(
      (['new', 'in_progress'] as const).map((status) => apiClient.getRecommendationsPage({ status, limit: 500 })),
    );
    return new Set(pages.flatMap((p) => p.items.map((r) => r.plant_id)));
  }, ['openActionPlantIds']);
}

interface NoActionPlantsProps {
  plants: Plant[];
  onViewPlant: (id: number) => void;
}

/** Pots the rules produced no open action for (usually rated healthy with every sensor in range), so every pot is visible. */
export const NoActionPlants: React.FC<NoActionPlantsProps> = ({ plants, onViewPlant }) => {
  const openIds = useOpenActionPlantIds().data;
  if (!openIds || plants.length === 0) return null;
  const quiet = plants.filter((p) => !openIds.has(p.id));
  if (quiet.length === 0) return null;

  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card" data-testid="no-action-plants">
      <div className="flex items-center gap-2 mb-1">
        <CheckCircle2 className="w-4 h-4 text-[#2E8B57]" aria-hidden="true" />
        <h2 className="text-sm font-bold text-[#1C2B23]">No action needed</h2>
        <span className="text-xs text-[#6B786E]">
          {quiet.length} of {plants.length} pots
        </span>
      </div>
      <p className="text-xs text-[#6B786E] mb-3">
        These pots have no open recommendations: the model rates their recent readings as fine with every sensor in range, or their actions were completed or dismissed.
      </p>
      <div className="flex flex-wrap gap-2">
        {quiet.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => onViewPlant(p.id)}
            className="text-xs px-3 py-1.5 rounded-lg border border-[#DDE8D8] bg-[#EDF4EB] text-[#173F32] hover:border-[#2E8B57]"
          >
            {p.name}
          </button>
        ))}
      </div>
    </div>
  );
};
