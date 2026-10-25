import React from 'react';
import { TIME_RANGE_LABELS, type TimeRange } from './format';

const RANGES: TimeRange[] = ['24h', '7d', 'all'];

interface TimeRangeButtonsProps {
  value: TimeRange;
  onChange: (range: TimeRange) => void;
}

/** 24 h / 7 days / full dataset toggle (ranges are relative to the dataset's latest reading). */
export const TimeRangeButtons: React.FC<TimeRangeButtonsProps> = ({ value, onChange }) => (
  <div className="flex items-center gap-1 bg-white border border-[#DCE4D9] rounded-lg p-1 text-xs">
    {RANGES.map((r) => (
      <button
        key={r}
        type="button"
        aria-pressed={value === r}
        onClick={() => onChange(r)}
        className={`px-2.5 py-1 rounded ${value === r ? 'bg-[#EDF4EB] text-[#2E8B57] font-bold' : 'text-[#6B786E]'}`}
      >
        {TIME_RANGE_LABELS[r]}
      </button>
    ))}
  </div>
);
