import React from 'react';
import { cn } from '../../lib/utils';

export interface CountItem {
  key: string;
  label: string;
  count: number;
  /** Optional trailing element, e.g. a small badge. */
  extra?: React.ReactNode;
  /** Overrides the bar colour for this item. */
  barClassName?: string;
}

interface CountBarsProps {
  items: CountItem[];
  /** Denominator for the bar width. Defaults to the sum of the counts. */
  total?: number;
  unit?: string;
  barClassName?: string;
  className?: string;
}

/** A list of labelled counts with a bar scaled to the share of `total`. Widths are data-driven, never fixed. */
export const CountBars: React.FC<CountBarsProps> = ({ items, total, unit = '', barClassName = 'bg-[#2E8B57]', className }) => {
  const denom = total ?? items.reduce((n, i) => n + i.count, 0);
  return (
    <div className={cn('grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3', className)}>
      {items.map((item) => {
        const pct = denom > 0 ? (item.count / denom) * 100 : 0;
        return (
          <div key={item.key} className="p-3.5 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8]">
            <div className="flex items-center justify-between gap-2 text-xs mb-1 text-[#173F32] font-semibold">
              <span className="inline-flex items-center gap-1">
                {item.label}
                {item.extra}
              </span>
              <span className="font-mono">
                {item.count.toLocaleString()}
                {unit ? ` ${unit}` : ''}
              </span>
            </div>
            <div className="w-full bg-[#DDE8D8] h-2 rounded-full overflow-hidden" role="presentation">
              <div className={cn('h-full', item.barClassName ?? barClassName)} style={{ width: `${pct}%` }} />
            </div>
            <div className="text-[10px] text-[#6B786E] mt-1 font-mono">{pct.toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
};
