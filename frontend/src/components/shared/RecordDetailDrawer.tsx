import React, { useState } from 'react';
import { X, Check, Copy, ShieldCheck, AlertTriangle, Clock, Radio, Cpu, Layers } from 'lucide-react';
import { SensorRecord } from '../../types';
import { StatusBadge } from './StatusBadge';
import { formatDateTime } from '../../lib/utils';

interface RecordDetailDrawerProps {
  record: SensorRecord | null;
  onClose: () => void;
}

export const RecordDetailDrawer: React.FC<RecordDetailDrawerProps> = ({ record, onClose }) => {
  const [copied, setCopied] = useState(false);

  if (!record) return null;

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(record, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/40 backdrop-blur-sm flex justify-end animate-fadeIn">
      <div className="w-full max-w-xl bg-white h-full shadow-2xl flex flex-col border-l border-[#DCE4D9] animate-slideIn">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[#DCE4D9] flex items-center justify-between bg-[#F6F7F0]">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
                Sensor Telemetry Record
              </span>
              <StatusBadge status={record.status || 'accepted'} size="sm" />
            </div>
            <h2 className="text-lg font-bold text-[#1C2B23]">
              Pot #{record.plant_id} • Session #{record.session_id}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-stone-500 hover:text-stone-800 hover:bg-stone-200/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Validation Alert */}
          {record.status === 'flagged' && (
            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-bold mb-0.5">Threshold Alert</p>
                <p className="text-amber-800 leading-relaxed">
                  {record.validation_error || 'Telemetry value crossed stress threshold boundaries.'}
                </p>
              </div>
            </div>
          )}

          {/* Simple Hardware Mapping Card */}
          <div className="bg-[#EDF4EB]/70 rounded-xl p-4 border border-[#DDE8D8]">
            <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#2E8B57]" /> Physical Hardware Routing
            </h4>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Plant / Pot:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Pot #{record.plant_id}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">ESP32 Node:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Node {record.node_id}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Multiplexer Channel:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Sampling Point {record.sampling_point}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Collection Pass:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Session #{record.session_id}</span>
              </div>
            </div>
          </div>

          {/* Calibrated Sensor Measurements */}
          <div>
            <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-3">
              Calibrated Physical Measurements
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Soil Moisture</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{record.soil.toFixed(1)}%</span>
                <span className="text-[10px] text-stone-500 block mt-0.5">Capacitive ADC probe</span>
              </div>
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Air Temperature</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{record.temperature.toFixed(1)}°C</span>
                <span className="text-[10px] text-stone-500 block mt-0.5">Ambient sensor</span>
              </div>
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Relative Humidity</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{record.humidity.toFixed(1)}%</span>
                <span className="text-[10px] text-stone-500 block mt-0.5">Air hygrometer</span>
              </div>
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Light Intensity</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{Math.round(record.light).toLocaleString()} lx</span>
                <span className="text-[10px] text-stone-500 block mt-0.5">Photodiode sensor</span>
              </div>
            </div>
          </div>

          {/* Timestamps */}
          <div className="space-y-2 border-t border-[#DCE4D9] pt-4">
            <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#426B4F]" /> Acquisition Timestamps
            </h4>
            <div className="text-xs space-y-2 bg-stone-50 p-3.5 rounded-xl border border-stone-200 font-mono">
              <div className="flex justify-between">
                <span className="text-[#6B786E]">Timestamp (ISO 8601):</span>
                <span className="text-[#1C2B23] font-bold">{record.timestamp}</span>
              </div>
              {record.received_at && (
                <div className="flex justify-between">
                  <span className="text-[#6B786E]">Jetson Received Time:</span>
                  <span className="text-[#1C2B23]">{record.received_at}</span>
                </div>
              )}
            </div>
          </div>

          {/* Raw JSON Record */}
          <div className="border-t border-[#DCE4D9] pt-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider">
                Raw JSON Schema Payload
              </h4>
              <button
                onClick={copyJson}
                className="text-xs font-semibold text-[#2E8B57] hover:text-[#173F32] flex items-center gap-1 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy JSON'}</span>
              </button>
            </div>
            <pre className="bg-[#102F26] text-[#DDE8D8] p-3.5 rounded-xl text-xs font-mono overflow-x-auto max-h-48 leading-relaxed">
              {JSON.stringify(record, null, 2)}
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[#DCE4D9] bg-[#F6F7F0] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-stone-200 hover:bg-stone-300 text-stone-800 text-xs font-semibold rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
