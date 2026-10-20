import React from 'react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';

interface LiveSparklineProps {
  data: number[];
  color?: string;
  height?: number;
  width?: number | string;
}

export const LiveSparkline: React.FC<LiveSparklineProps> = ({
  data,
  color = '#2E8B57',
  height = 36,
  width = '100%',
}) => {
  const chartData = data.map((val, idx) => ({ index: idx, value: val }));

  return (
    <div style={{ width, height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
