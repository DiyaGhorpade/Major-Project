import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceArea,
} from 'recharts';
import { SensorRecord } from '../../types';
import { formatDateTime } from '../../lib/utils';

interface SensorTrendChartProps {
  records: SensorRecord[];
  title?: string;
  defaultSensor?: 'all' | 'soil' | 'temperature' | 'humidity' | 'light';
  optimalRange?: {
    soil?: { min: number; max: number };
    temperature?: { min: number; max: number };
    humidity?: { min: number; max: number };
    light?: { min: number; max: number };
  };
  height?: number;
}

export const SensorTrendChart: React.FC<SensorTrendChartProps> = ({
  records,
  title,
  defaultSensor = 'all',
  optimalRange,
  height = 320,
}) => {
  const [selectedSensor, setSelectedSensor] = useState<'all' | 'soil' | 'temperature' | 'humidity' | 'light'>(
    defaultSensor
  );

  // Prepare data sorted chronologically
  const chartData = [...records]
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    .map((r) => {
      const date = new Date(r.timestamp);
      return {
        timestamp: r.timestamp,
        timeLabel: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }),
        fullDate: formatDateTime(r.timestamp),
        soil: r.soil,
        temperature: r.temperature,
        humidity: r.humidity,
        light: r.light,
        plantId: r.plant_id,
        samplingPoint: r.sampling_point,
      };
    });

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      return (
        <div className="bg-[#102F26] text-white p-3 rounded-lg shadow-xl border border-[#2E8B57]/40 text-xs space-y-1.5">
          <div className="font-semibold text-[#DDE8D8] border-b border-white/10 pb-1">
            {dataPoint.fullDate}
          </div>
          {payload.map((entry: any, index: number) => (
            <div key={`item-${index}`} className="flex items-center justify-between gap-4">
              <span className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block"
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-[#EDF4EB] capitalize">{entry.name}:</span>
              </span>
              <span className="font-mono font-bold text-white">
                {entry.value} {entry.name === 'soil' || entry.name === 'humidity' ? '%' : entry.name === 'temperature' ? '°C' : 'lx'}
              </span>
            </div>
          ))}
          {dataPoint.plantId && (
            <div className="text-[10px] text-stone-400 pt-1 border-t border-white/10">
              Plant: {dataPoint.plantId.toUpperCase()} ({dataPoint.samplingPoint})
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        {title && <h3 className="text-sm font-bold text-[#1C2B23]">{title}</h3>}
        
        {/* Sensor selector tabs */}
        <div className="inline-flex rounded-lg bg-[#EDF4EB] p-1 text-xs font-medium text-[#173F32]">
          <button
            onClick={() => setSelectedSensor('all')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              selectedSensor === 'all'
                ? 'bg-white text-[#102F26] shadow-sm font-semibold'
                : 'hover:text-[#102F26]'
            }`}
          >
            All Metrics
          </button>
          <button
            onClick={() => setSelectedSensor('soil')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              selectedSensor === 'soil'
                ? 'bg-white text-[#2E8B57] shadow-sm font-semibold'
                : 'hover:text-[#102F26]'
            }`}
          >
            Soil Moisture (%)
          </button>
          <button
            onClick={() => setSelectedSensor('temperature')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              selectedSensor === 'temperature'
                ? 'bg-white text-[#C98227] shadow-sm font-semibold'
                : 'hover:text-[#102F26]'
            }`}
          >
            Temp (°C)
          </button>
          <button
            onClick={() => setSelectedSensor('humidity')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              selectedSensor === 'humidity'
                ? 'bg-white text-[#3977B8] shadow-sm font-semibold'
                : 'hover:text-[#102F26]'
            }`}
          >
            Humidity (%)
          </button>
          <button
            onClick={() => setSelectedSensor('light')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              selectedSensor === 'light'
                ? 'bg-white text-[#D4A338] shadow-sm font-semibold'
                : 'hover:text-[#102F26]'
            }`}
          >
            Light (lx)
          </button>
        </div>
      </div>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#EDF4EB" vertical={false} />
            <XAxis
              dataKey="timeLabel"
              stroke="#6B786E"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#DCE4D9' }}
            />
            
            {/* Left Y Axis for percentages and temperatures */}
            <YAxis
              yAxisId="left"
              stroke="#6B786E"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#DCE4D9' }}
              domain={selectedSensor === 'light' ? ['auto', 'auto'] : [0, 100]}
              unit={selectedSensor === 'temperature' ? '°C' : selectedSensor === 'light' ? ' lx' : '%'}
            />

            {/* Right Y Axis if viewing all metrics (for Light in lux) */}
            {selectedSensor === 'all' && (
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#D4A338"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#DCE4D9' }}
                unit=" lx"
                domain={[0, 8000]}
              />
            )}

            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }}
              iconType="circle"
            />

            {/* Optimal Range Reference Band if looking at single sensor */}
            {selectedSensor === 'soil' && optimalRange?.soil && (
              <ReferenceArea
                yAxisId="left"
                y1={optimalRange.soil.min}
                y2={optimalRange.soil.max}
                fill="#2E8B57"
                fillOpacity={0.12}
                stroke="#2E8B57"
                strokeDasharray="3 3"
              />
            )}
            {selectedSensor === 'temperature' && optimalRange?.temperature && (
              <ReferenceArea
                yAxisId="left"
                y1={optimalRange.temperature.min}
                y2={optimalRange.temperature.max}
                fill="#C98227"
                fillOpacity={0.12}
                stroke="#C98227"
                strokeDasharray="3 3"
              />
            )}
            {selectedSensor === 'humidity' && optimalRange?.humidity && (
              <ReferenceArea
                yAxisId="left"
                y1={optimalRange.humidity.min}
                y2={optimalRange.humidity.max}
                fill="#3977B8"
                fillOpacity={0.12}
                stroke="#3977B8"
                strokeDasharray="3 3"
              />
            )}

            {(selectedSensor === 'all' || selectedSensor === 'soil') && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="soil"
                name="Soil Moisture"
                stroke="#2E8B57"
                strokeWidth={2.5}
                dot={{ r: 3, fill: '#2E8B57' }}
                activeDot={{ r: 6 }}
              />
            )}

            {(selectedSensor === 'all' || selectedSensor === 'temperature') && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="temperature"
                name="Temperature"
                stroke="#C98227"
                strokeWidth={2}
                dot={{ r: 3, fill: '#C98227' }}
                activeDot={{ r: 6 }}
              />
            )}

            {(selectedSensor === 'all' || selectedSensor === 'humidity') && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="humidity"
                name="Humidity"
                stroke="#3977B8"
                strokeWidth={2}
                dot={{ r: 3, fill: '#3977B8' }}
                activeDot={{ r: 6 }}
              />
            )}

            {(selectedSensor === 'all' || selectedSensor === 'light') && (
              <Line
                yAxisId={selectedSensor === 'all' ? 'right' : 'left'}
                type="monotone"
                dataKey="light"
                name="Light"
                stroke="#D4A338"
                strokeWidth={1.5}
                strokeDasharray={selectedSensor === 'all' ? '4 4' : undefined}
                dot={{ r: 2, fill: '#D4A338' }}
                activeDot={{ r: 5 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
