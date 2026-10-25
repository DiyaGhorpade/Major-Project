import React from 'react';
import { Droplets, Thermometer, Wind, Sun, AlertCircle, CheckCircle2 } from 'lucide-react';
import { cn } from '../../lib/utils';

export type SensorType = 'soil' | 'temperature' | 'humidity' | 'light';

interface SensorReadingCardProps {
  type: SensorType;
  value: number | null | undefined;
  optimalRange?: {
    min: number;
    max: number;
    unit: string;
    lowLabel?: string;
    idealLabel?: string;
    highLabel?: string;
  };
  timestamp?: string;
  isStale?: boolean;
  className?: string;
}

export const SensorReadingCard: React.FC<SensorReadingCardProps> = ({
  type,
  value,
  optimalRange,
  timestamp,
  isStale = false,
  className,
}) => {
  const configs = {
    soil: {
      title: 'Soil Moisture',
      simpleName: 'Soil Water Level',
      unit: '%',
      icon: <Droplets className="w-4 h-4 text-emerald-600" />,
      minBound: 0,
      maxBound: 100,
      lowText: 'Thirsty / Dry',
      idealText: 'Ideal Moisture',
      highText: 'Overwatered',
    },
    temperature: {
      title: 'Air Temperature',
      simpleName: 'Room Temperature',
      unit: '°C',
      icon: <Thermometer className="w-4 h-4 text-amber-600" />,
      minBound: null,
      maxBound: null,
      lowText: 'Chilly',
      idealText: 'Warm & Cosy',
      highText: 'Too Hot',
    },
    humidity: {
      title: 'Air Humidity',
      simpleName: 'Air Moisture',
      unit: '%',
      icon: <Wind className="w-4 h-4 text-sky-600" />,
      minBound: 0,
      maxBound: 100,
      lowText: 'Dry Air',
      idealText: 'Pleasant Air',
      highText: 'Very Humid',
    },
    light: {
      title: 'Light Level',
      simpleName: 'Sunlight / Photoperiod',
      unit: 'lux',
      icon: <Sun className="w-4 h-4 text-yellow-600" />,
      minBound: null,
      maxBound: null,
      lowText: 'Shaded / Dim',
      idealText: 'Sunny & Bright',
      highText: 'Blinding',
    },
  };

  const cfg = configs[type];
  const isUnconfigured = value === null || value === undefined;

  let status: 'optimal' | 'low' | 'high' | 'unconfigured' = 'optimal';
  let percentage = 50;
  // Zone widths follow the real optimal range on the gauge scale (not a fixed 30/45/25 split).
  let lowWidth = 30;
  let idealWidth = 45;

  if (isUnconfigured) {
    status = 'unconfigured';
  } else if (optimalRange) {
    if (value < optimalRange.min) {
      status = 'low';
    } else if (value > optimalRange.max) {
      status = 'high';
    } else {
      status = 'optimal';
    }

    // Percent sensors use 0-100. Temperature and light have no natural bounds, so the gauge scale is the plant's
    // optimal range plus half its width on each side (never below 0), instead of a fixed number.
    const half = (optimalRange.max - optimalRange.min) / 2;
    const minB = cfg.minBound ?? Math.max(0, optimalRange.min - half);
    const maxB = cfg.maxBound ?? optimalRange.max + half;
    const span = maxB - minB || 1;
    const pct = (v: number) => Math.max(0, Math.min(100, ((v - minB) / span) * 100));
    percentage = Math.max(3, Math.min(97, pct(value)));
    lowWidth = pct(optimalRange.min);
    idealWidth = Math.max(0, pct(optimalRange.max) - lowWidth);
  }

  return (
    <div
      className={cn(
        'bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-col justify-between hover:shadow-card-hover transition-all',
        isStale && 'opacity-75 border-dashed',
        status === 'low' && 'border-amber-300 bg-amber-50/20',
        className
      )}
    >
      {/* Header with Title and Layman Friendly Badge */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-[#EDF4EB]">{cfg.icon}</div>
          <div>
            <span className="text-xs font-bold text-[#1C2B23] block leading-tight">{cfg.title}</span>
            <span className="text-[10px] text-[#6B786E]">{cfg.simpleName}</span>
          </div>
        </div>

        {status === 'optimal' && !isUnconfigured && (
          <span className="text-[11px] font-semibold text-[#2E8B57] bg-[#EDF4EB] px-2.5 py-1 rounded-full flex items-center gap-1 border border-[#DDE8D8]">
            <CheckCircle2 className="w-3 h-3" /> {cfg.idealText}
          </span>
        )}
        {status === 'low' && (
          <span className="text-[11px] font-bold text-amber-800 bg-amber-100 px-2.5 py-1 rounded-full flex items-center gap-1 border border-amber-300 animate-pulse">
            <AlertCircle className="w-3 h-3 text-amber-600" /> {cfg.lowText}
          </span>
        )}
        {status === 'high' && (
          <span className="text-[11px] font-bold text-red-800 bg-red-100 px-2.5 py-1 rounded-full flex items-center gap-1 border border-red-300">
            <AlertCircle className="w-3 h-3 text-red-600" /> {cfg.highText}
          </span>
        )}
        {status === 'unconfigured' && (
          <span className="text-[11px] font-medium text-stone-500 bg-stone-100 px-2 py-0.5 rounded-full">
            No value
          </span>
        )}
      </div>

      {/* Main Big Metric Display */}
      <div className="my-2">
        {isUnconfigured ? (
          <div className="text-xs font-medium text-stone-400 py-1">
            n/a: this reading has no value for this sensor
          </div>
        ) : (
          <div className="flex items-baseline gap-1.5">
            <span className="text-3xl font-extrabold tracking-tight text-[#1C2B23]">
              {type === 'light' ? Math.round(value).toLocaleString() : value.toFixed(1)}
            </span>
            <span className="text-sm font-bold text-[#6B786E]">{cfg.unit}</span>
          </div>
        )}
      </div>

      {/* Visual Zone Gauge Bar */}
      {!isUnconfigured && optimalRange && (
        <div className="space-y-1.5 pt-2 border-t border-[#DCE4D9]">
          <div className="relative">
            {/* Visual segmented zone bar */}
            <div className="w-full h-2 rounded-full overflow-hidden flex bg-stone-100">
              <div className="bg-amber-200" style={{ width: `${lowWidth}%` }} title="Low zone" />
              <div className="bg-[#79B86A]" style={{ width: `${idealWidth}%` }} title="Ideal zone" />
              <div className="bg-red-200" style={{ width: `${Math.max(0, 100 - lowWidth - idealWidth)}%` }} title="High zone" />
            </div>

            {/* Current value marker pin */}
            <div
              className="absolute -top-1 w-3 h-4 -ml-1.5 flex flex-col items-center transition-all duration-300"
              style={{ left: `${percentage}%` }}
            >
              <div className="w-2.5 h-2.5 rounded-full bg-[#102F26] border-2 border-white shadow-sm" />
            </div>
          </div>

          <div className="flex justify-between items-center text-[10px] text-[#6B786E] pt-0.5">
            <span>Ideal for Tulsi: <strong className="text-[#102F26]">{optimalRange.min}–{optimalRange.max} {optimalRange.unit}</strong></span>
            {timestamp && <span className="text-stone-400">{timestamp}</span>}
          </div>
        </div>
      )}
    </div>
  );
};
