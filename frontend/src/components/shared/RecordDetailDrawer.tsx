import React, { useEffect, useState } from 'react';
import { X, Check, Copy, AlertTriangle, Clock, Cpu, Sparkles } from 'lucide-react';
import { apiClient } from '../../api/client';
import { SensorRecord } from '../../types';
import { useAsync } from '../../hooks/useApi';
import { StatusBadge } from './StatusBadge';
import { AnomalyFlag, EnvStateLabel, SeverityBadge } from '../monitoring/PredictionBadges';
import { ProbabilityBars } from '../monitoring/ProbabilityBars';
import { DatasetLabelsBlock } from '../monitoring/DatasetLabelsBlock';
import { formatConfidence } from '../monitoring/format';
import { InfoTip } from './Disclaimers';
import { DISCLAIMERS } from './disclaimerText';
import { formatPercent, formatTemp, formatLux, formatDateTime, orNA } from '../../lib/utils';

interface RecordDetailDrawerProps {
  record: SensorRecord | null;
  onClose: () => void;
}

/** ML output for exactly this stored reading (matched by record_id); null once loaded means no Run yet. */
const RecordPrediction: React.FC<{ record: SensorRecord }> = ({ record }) => {
  const query = useAsync(
    async () => {
      const page = await apiClient.getPredictionsPage({
        plantId: record.plant_id,
        from: record.timestamp,
        to: record.timestamp,
        limit: 5,
      });
      return page.items.find((p) => p.record_id === record.record_id) ?? null;
    },
    ['record-prediction', record.record_id],
  );
  const pred = query.data;

  return (
    <div data-testid="record-ml" className="rounded-xl border border-[#DDE8D8] bg-[#EDF4EB]/50 p-3.5">
      <h4 className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[#1C2B23]">
        <Sparkles className="h-3.5 w-3.5 text-[#2E8B57]" aria-hidden="true" />
        Model output (ML)
      </h4>
      {query.loading ? (
        <p className="text-xs text-stone-500">Loading model output...</p>
      ) : query.error ? (
        <p className="text-xs text-red-700">Could not load model output: {query.error.message}</p>
      ) : !pred || pred.status !== 'ok' ? (
        <p className="text-xs text-stone-500">
          {pred ? `No model output for this reading (${pred.status}${pred.error ? `: ${pred.error}` : ''}).` : `No model output for this reading yet. ${DISCLAIMERS.runPrompt}`}
        </p>
      ) : (
        <div className="space-y-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="text-stone-500">Severity</span>
            <span className="flex items-center gap-2">
              <SeverityBadge severity={pred.severity} />
              <span className="font-mono text-stone-600">{formatConfidence(pred.confidence)}</span>
            </span>
          </div>
          <ProbabilityBars probabilities={pred.probabilities} />
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1 text-stone-500">
              Stress type guess <InfoTip text={DISCLAIMERS.conditionGuess} />
            </span>
            <span className="font-semibold text-[#1C2B23]">{pred.condition ?? 'n/a'}</span>
          </div>
          <div className="text-stone-600">
            <EnvStateLabel envState={pred.envState} />
          </div>
          <div>
            <AnomalyFlag isAnomaly={pred.anomaly?.isAnomaly} />
          </div>
          <p className="text-[10px] text-stone-500">Model {orNA(pred.modelVersion)}</p>
        </div>
      )}
    </div>
  );
};

const DrawerBody: React.FC<{ record: SensorRecord; onClose: () => void }> = ({ record, onClose }) => {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');

  // Reset the button label after a moment; the timer is cleared if the drawer closes first.
  useEffect(() => {
    if (copyState === 'idle') return;
    const id = setTimeout(() => setCopyState('idle'), 2000);
    return () => clearTimeout(id);
  }, [copyState]);

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(record, null, 2));
      setCopyState('copied');
    } catch {
      setCopyState('failed'); // clipboard unavailable or permission denied
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/40 backdrop-blur-sm flex justify-end animate-fadeIn" role="dialog" aria-label="Sensor record details">
      <div className="w-full max-w-xl bg-white h-full shadow-2xl flex flex-col border-l border-[#DCE4D9] animate-slideIn">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[#DCE4D9] flex items-center justify-between bg-[#F6F7F0]">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">Sensor Telemetry Record</span>
              <StatusBadge status={record.status || 'accepted'} size="sm" />
            </div>
            <h2 className="text-lg font-bold text-[#1C2B23]">
              Pot #{record.plant_id} • Session {record.session_id === null ? 'n/a' : `#${record.session_id}`}
            </h2>
            <p className="text-[11px] font-mono text-stone-500">{record.record_id}</p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close record details"
            className="p-2 rounded-lg text-stone-500 hover:text-stone-800 hover:bg-stone-200/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {(record.status === 'flagged' || record.status === 'rejected' || record.validation_error) && (
            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-bold mb-0.5">Validation: {record.status}</p>
                <p className="text-amber-800 leading-relaxed" data-testid="validation-error">
                  {record.validation_error ?? 'No validation message recorded.'}
                </p>
              </div>
            </div>
          )}

          <div className="bg-[#EDF4EB]/70 rounded-xl p-4 border border-[#DDE8D8]">
            <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#2E8B57]" /> Source Mapping
            </h4>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Plant / Pot:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Pot #{record.plant_id}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Sensor Node:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Node {record.node_id}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Sampling Point:</span>
                <span className="font-bold text-[#1C2B23] text-sm">{orNA(record.sampling_point)}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-[#DDE8D8]">
                <span className="text-[#6B786E] block mb-0.5 text-[11px]">Collection Pass:</span>
                <span className="font-bold text-[#1C2B23] text-sm">Session {record.session_id === null ? 'n/a' : `#${record.session_id}`}</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-3">Sensor Measurements</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Soil Moisture</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{formatPercent(record.soil)}</span>
              </div>
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Air Temperature</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{formatTemp(record.temperature)}</span>
              </div>
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Relative Humidity</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{formatPercent(record.humidity)}</span>
              </div>
              <div className="p-3.5 rounded-xl border border-[#DCE4D9] bg-white">
                <span className="text-xs text-[#6B786E] block">Light Intensity</span>
                <span className="text-2xl font-bold text-[#1C2B23]">{formatLux(record.light)}</span>
              </div>
            </div>
          </div>

          <RecordPrediction record={record} />

          <DatasetLabelsBlock labels={record.datasetLabels} />

          <div className="space-y-2 border-t border-[#DCE4D9] pt-4">
            <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#426B4F]" /> Timestamps
            </h4>
            <div className="text-xs space-y-2 bg-stone-50 p-3.5 rounded-xl border border-stone-200 font-mono">
              <div className="flex justify-between gap-3">
                <span className="text-[#6B786E]">Timestamp (ISO 8601):</span>
                <span className="text-[#1C2B23] font-bold">{record.timestamp}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-[#6B786E]">Timestamp (IST):</span>
                <span className="text-[#1C2B23]">{formatDateTime(record.timestamp)}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-[#6B786E]">Received:</span>
                <span className="text-[#1C2B23]">{orNA(record.received_at)}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-[#6B786E]">Signal (RSSI) / CRC:</span>
                <span className="text-[#1C2B23]">
                  {orNA(record.rssi)} / {record.crc_ok === null ? 'n/a' : record.crc_ok ? 'ok' : 'failed'}
                </span>
              </div>
            </div>
          </div>

          <div className="border-t border-[#DCE4D9] pt-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider">Raw JSON Payload</h4>
              <button onClick={() => void copyJson()} className="text-xs font-semibold text-[#2E8B57] hover:text-[#173F32] flex items-center gap-1 transition-colors">
                {copyState === 'copied' ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copyState === 'copied' ? 'Copied' : copyState === 'failed' ? 'Copy failed' : 'Copy JSON'}</span>
              </button>
            </div>
            <pre className="bg-[#102F26] text-[#DDE8D8] p-3.5 rounded-xl text-xs font-mono overflow-x-auto max-h-48 leading-relaxed">
              {JSON.stringify(record, null, 2)}
            </pre>
          </div>
        </div>

        <div className="px-6 py-4 border-t border-[#DCE4D9] bg-[#F6F7F0] flex justify-end">
          <button onClick={onClose} className="px-4 py-2 bg-stone-200 hover:bg-stone-300 text-stone-800 text-xs font-semibold rounded-lg transition-colors">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export const RecordDetailDrawer: React.FC<RecordDetailDrawerProps> = ({ record, onClose }) => {
  if (!record) return null;
  return <DrawerBody record={record} onClose={onClose} />;
};
