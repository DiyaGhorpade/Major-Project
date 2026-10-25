import React from 'react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';

interface LiveSparklineProps {
  /** Oldest first. Null values (missing sensor readings) break the line instead of being drawn as 0. */
  data: ReadonlyArray<number | null>;
  color?: string;
  height?: number;
  width?: number | string;
}

export const LiveSparkline: React.FC<LiveSparklineProps> = ({ data, color = '#2E8B57', height = 36, width = '100%' }) => {
  const hasValue = data.some((v) => v !== null);
  if (!hasValue) {
    return (
      <div style={{ width, height }} className="flex items-center justify-end text-[10px] text-stone-400">
        n/a
      </div>
    );
  }
  const chartData = data.map((value, index) => ({ index, value }));

  return (
    <div style={{ width, height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={data.length < 3}
            connectNulls={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
