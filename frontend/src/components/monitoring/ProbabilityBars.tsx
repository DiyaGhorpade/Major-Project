import React from 'react';

const ORDER = ['healthy', 'low', 'moderate', 'high'];

interface ProbabilityBarsProps {
  probabilities: Record<string, number> | null | undefined;
}

/** Severity class probabilities from the model (ML output), in a fixed order. Null-safe. */
export const ProbabilityBars: React.FC<ProbabilityBarsProps> = ({ probabilities }) => {
  if (!probabilities) return <span className="text-xs text-stone-500">Probabilities: n/a</span>;
  const keys = [...ORDER.filter((k) => k in probabilities), ...Object.keys(probabilities).filter((k) => !ORDER.includes(k))];
  return (
    <div className="space-y-1" data-testid="probability-bars">
      {keys.map((k) => {
        const p = probabilities[k] ?? 0;
        return (
          <div key={k} className="flex items-center gap-2 text-[11px]">
            <span className="w-16 capitalize text-stone-600">{k}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-stone-200">
              <div className="h-full rounded-full bg-[#2E8B57]" style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }} />
            </div>
            <span className="w-10 text-right font-mono text-stone-700">{(p * 100).toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );
};
