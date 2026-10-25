import React, { useMemo, useState } from 'react';
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
import { formatDateTime } from '../../lib/utils';
import { formatIstTick, toMillis } from '../monitoring/format';

/** Anything with a timestamp and the four sensor values (a stored reading or a monitoring series point). */
export interface TrendPoint {
  timestamp: string;
  soil: number | null;
  temperature: number | null;
  humidity: number | null;
  light: number | null;
  plant_id?: number;
  sampling_point?: number | null;
}

type Sensor = 'soil' | 'temperature' | 'humidity' | 'light';
type Selected = 'all' | Sensor;

interface SensorTrendChartProps {
  records: readonly TrendPoint[];
  title?: string;
  defaultSensor?: Selected;
  optimalRange?: Partial<Record<Sensor, { min: number; max: number }>>;
  height?: number;
}

interface ChartRow {
  ts: number;
  fullDate: string;
  soil: number | null;
  temperature: number | null;
  humidity: number | null;
  light: number | null;
  plantId: number | null;
  samplingPoint: number | null;
}

const SENSORS: Record<Sensor, { name: string; unit: string; color: string }> = {
  soil: { name: 'Soil Moisture', unit: '%', color: '#2E8B57' },
  temperature: { name: 'Temperature', unit: '°C', color: '#C98227' },
  humidity: { name: 'Humidity', unit: '%', color: '#3977B8' },
  light: { name: 'Light', unit: 'lx', color: '#D4A338' },
};

const TABS: Array<{ key: Selected; label: string }> = [
  { key: 'all', label: 'All Metrics' },
  { key: 'soil', label: 'Soil Moisture (%)' },
  { key: 'temperature', label: 'Temp (°C)' },
  { key: 'humidity', label: 'Humidity (%)' },
  { key: 'light', label: 'Light (lx)' },
];

interface TooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload: ChartRow }>;
  visible: Sensor[];
}

const ChartTooltip: React.FC<TooltipProps> = ({ active, payload, visible }) => {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="bg-[#102F26] text-white p-3 rounded-lg shadow-xl border border-[#2E8B57]/40 text-xs space-y-1.5">
      <div className="font-semibold text-[#DDE8D8] border-b border-white/10 pb-1">{row.fullDate} IST</div>
      {visible.map((key) => {
        const value = row[key];
        return (
          <div key={key} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: SENSORS[key].color }} />
              <span className="text-[#EDF4EB]">{SENSORS[key].name}:</span>
            </span>
            <span className="font-mono font-bold text-white">
              {value === null ? 'n/a' : `${key === 'light' ? Math.round(value).toLocaleString() : value} ${SENSORS[key].unit}`}
            </span>
          </div>
        );
      })}
      {row.plantId !== null && (
        <div className="text-[10px] text-stone-400 pt-1 border-t border-white/10">
          Pot #{row.plantId}
          {row.samplingPoint !== null ? ` (sampling point ${row.samplingPoint})` : ''}
        </div>
      )}
    </div>
  );
};

export const SensorTrendChart: React.FC<SensorTrendChartProps> = ({
  records,
  title,
  defaultSensor = 'all',
  optimalRange,
  height = 320,
}) => {
  const [selectedSensor, setSelectedSensor] = useState<Selected>(defaultSensor);

  // Chronological rows on a numeric time axis. Missing values stay null so the line breaks (never drawn as 0).
  const chartData = useMemo<ChartRow[]>(
    () =>
      records
        .map((r) => ({ r, ts: toMillis(r.timestamp) }))
        .filter((x): x is { r: TrendPoint; ts: number } => x.ts !== null)
        .sort((a, b) => a.ts - b.ts)
        .map(({ r, ts }) => ({
          ts,
          fullDate: formatDateTime(r.timestamp),
          soil: r.soil,
          temperature: r.temperature,
          humidity: r.humidity,
          light: r.light,
          plantId: r.plant_id ?? null,
          samplingPoint: r.sampling_point ?? null,
        })),
    [records],
  );

  const spansDays = chartData.length > 1 && chartData[chartData.length - 1].ts - chartData[0].ts > 36 * 3600 * 1000;
  const showDots = chartData.length <= 48;
  const visible: Sensor[] = selectedSensor === 'all' ? ['soil', 'temperature', 'humidity', 'light'] : [selectedSensor];
  const hasAnyValue = chartData.some((r) => visible.some((k) => r[k] !== null));

  const leftDomain: [number | 'auto', number | 'auto'] =
    selectedSensor === 'light' ? [0, 'auto'] : selectedSensor === 'temperature' ? ['auto', 'auto'] : [0, 100];
  const leftUnit = selectedSensor === 'temperature' ? '°C' : selectedSensor === 'light' ? ' lx' : '%';

  const band = selectedSensor !== 'all' ? optimalRange?.[selectedSensor] : undefined;

  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        {title && <h3 className="text-sm font-bold text-[#1C2B23]">{title}</h3>}

        <div className="inline-flex flex-wrap rounded-lg bg-[#EDF4EB] p-1 text-xs font-medium text-[#173F32]">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setSelectedSensor(tab.key)}
              className={`px-2.5 py-1 rounded-md transition-all ${
                selectedSensor === tab.key ? 'bg-white text-[#102F26] shadow-sm font-semibold' : 'hover:text-[#102F26]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {!hasAnyValue ? (
        <div style={{ height }} className="flex items-center justify-center text-xs text-stone-500" data-testid="trend-empty">
          No readings to chart for this selection.
        </div>
      ) : (
        <div style={{ width: '100%', height }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#EDF4EB" vertical={false} />
              <XAxis
                dataKey="ts"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(v: number) => formatIstTick(v, spansDays)}
                stroke="#6B786E"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#DCE4D9' }}
                minTickGap={40}
              />

              <YAxis
                yAxisId="left"
                stroke="#6B786E"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#DCE4D9' }}
                domain={leftDomain}
                unit={leftUnit}
              />

              {selectedSensor === 'all' && (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#D4A338"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#DCE4D9' }}
                  unit=" lx"
                  domain={[0, 'auto']}
                />
              )}

              <Tooltip content={<ChartTooltip visible={visible} />} />
              <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} iconType="circle" />

              {band && selectedSensor !== 'all' && (
                <ReferenceArea
                  yAxisId="left"
                  y1={band.min}
                  y2={band.max}
                  ifOverflow="hidden"
                  fill={SENSORS[selectedSensor].color}
                  fillOpacity={0.12}
                  stroke={SENSORS[selectedSensor].color}
                  strokeDasharray="3 3"
                />
              )}

              {visible.map((key) => (
                <Line
                  key={key}
                  yAxisId={key === 'light' && selectedSensor === 'all' ? 'right' : 'left'}
                  type="monotone"
                  dataKey={key}
                  name={SENSORS[key].name}
                  stroke={SENSORS[key].color}
                  strokeWidth={key === 'soil' ? 2.5 : key === 'light' ? 1.5 : 2}
                  strokeDasharray={key === 'light' && selectedSensor === 'all' ? '4 4' : undefined}
                  dot={showDots ? { r: key === 'light' ? 2 : 3, fill: SENSORS[key].color } : false}
                  activeDot={{ r: 5 }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <p className="mt-2 text-[10px] text-[#6B786E]">
        Times shown in IST. Missing sensor values appear as gaps, not zeros.
      </p>
    </div>
  );
};
