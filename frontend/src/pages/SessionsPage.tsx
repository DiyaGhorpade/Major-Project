import React, { useState, useEffect } from 'react';
import {
  History,
  Clock,
  ChevronRight,
  ShieldCheck,
  Cpu,
} from 'lucide-react';
import { useSessions } from '../hooks/useData';
import { Session } from '../types';
import { StatusBadge } from '../components/shared/StatusBadge';
import { formatDateTime } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface SessionsPageProps {
  onNavigate: (route: NavRoute) => void;
}

export const SessionsPage: React.FC<SessionsPageProps> = ({ onNavigate }) => {
  const sessionsQuery = useSessions();
  const sessions = sessionsQuery.data ?? [];
  const loading = sessionsQuery.loading;
  const [pickedId, setPickedId] = useState<number | null>(null);
  const selectedSession = sessions.find((s) => s.session_id === pickedId) ?? sessions[0] ?? null;
  const setSelectedSession = (s: Session | null) => setPickedId(s ? s.session_id : null);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              Ingestion Session Registry
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              3x Daily Protocol (08:00, 13:00, 18:00)
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            Historical Collection Runs & Run Audits
          </h1>
          <p className="text-xs text-[#6B786E] mt-1">
            Audit logs for sequential sampling passes across Node 1 (12 channels) and Node 2 (4 channels).
          </p>
        </div>

        <button
          onClick={() => onNavigate('data-logs')}
          className="px-3.5 py-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-semibold flex items-center gap-2 transition-colors self-start md:self-auto"
        >
          <History className="w-3.5 h-3.5" />
          <span>View All Log Records</span>
        </button>
      </div>

      {/* Main Grid: Sessions List + Selected Session Detail Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Session Table / List */}
        <div className="lg:col-span-2 bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
          <div className="p-4 border-b border-[#DCE4D9] flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#1C2B23]">Collection Passes</h3>
            <span className="text-xs font-mono text-stone-500">{sessions.length} runs logged</span>
          </div>

          <div className="divide-y divide-[#DCE4D9]/60">
            {sessions.map((s) => {
              const isSelected = selectedSession?.session_id === s.session_id;
              const isFull = s.received_records === s.expected_records;

              return (
                <div
                  key={s.session_id}
                  onClick={() => setSelectedSession(s)}
                  className={`p-4 cursor-pointer transition-colors flex items-center justify-between gap-3 ${
                    isSelected ? 'bg-[#EDF4EB]/70 border-l-4 border-[#2E8B57]' : 'hover:bg-stone-50'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-bold text-xs">
                      <Clock className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="font-mono text-xs font-bold text-[#1C2B23]">
                          {s.session_id}
                        </span>
                        <StatusBadge status={s.status} size="sm" />
                      </div>
                      <div className="text-[11px] text-[#6B786E]">
                        Started: {formatDateTime(s.started_at)} • Duration: {s.duration_seconds}s
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 text-right">
                    <div>
                      <div className="text-xs font-bold text-[#1C2B23]">
                        {s.received_records} / {s.expected_records}
                      </div>
                      <div className="text-[10px] text-stone-500 font-medium">
                        {s.rejected_records > 0 ? (
                          <span className="text-[#C94C4C] font-bold">
                            {s.rejected_records} rejected
                          </span>
                        ) : (
                          <span className="text-[#2E8B57]">100% accepted</span>
                        )}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-stone-400" />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Col: Selected Session Breakdown View */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col justify-between">
          {selectedSession ? (
            <div className="space-y-5">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-[#426B4F]">
                    Session Audit
                  </span>
                  <StatusBadge status={selectedSession.status} size="sm" />
                </div>
                <h3 className="text-base font-bold text-[#1C2B23] font-mono">
                  {selectedSession.session_id}
                </h3>
              </div>

              {/* Progress Summary */}
              <div className="p-3.5 rounded-xl bg-stone-50 border border-stone-200 text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-[#6B786E]">Scheduled Time:</span>
                  <span className="font-mono text-[#1C2B23]">{formatDateTime(selectedSession.scheduled_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#6B786E]">Execution Completed:</span>
                  <span className="font-mono text-[#1C2B23]">
                    {selectedSession.ended_at ? formatDateTime(selectedSession.ended_at) : 'In Progress'}
                  </span>
                </div>
                <div className="flex justify-between font-semibold border-t border-stone-200 pt-2">
                  <span className="text-[#6B786E]">Acquisition Duration:</span>
                  <span className="text-[#2E8B57]">{selectedSession.duration_seconds} seconds</span>
                </div>
              </div>

              {/* Node-by-Node Progress */}
              <div>
                <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-[#2E8B57]" /> Node Telemetry Breakdown
                </h4>
                <div className="space-y-2.5">
                  {Object.entries(selectedSession.node_progress).map(([nodeId, prog]) => (
                    <div
                      key={nodeId}
                      className="p-3 rounded-lg border border-[#DCE4D9] bg-white text-xs"
                    >
                      <div className="flex justify-between items-center mb-1 font-semibold text-[#1C2B23]">
                        <span className="uppercase">{nodeId} (Multiplexer)</span>
                        <span className="text-[#2E8B57]">
                          {prog.received} / {prog.expected} channels
                        </span>
                      </div>
                      <div className="w-full bg-[#EDF4EB] h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-[#2E8B57] h-full"
                          style={{ width: `${(prog.received / prog.expected) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Integrity Sign-off */}
              <div className="p-3 rounded-lg bg-[#EDF4EB] border border-[#DDE8D8] text-xs text-[#173F32] flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-[#2E8B57] shrink-0" />
                <span>Payload validated against CRC check and ADC dry/wet calibration points.</span>
              </div>
            </div>
          ) : (
            <div className="text-center text-xs text-stone-400 py-12">
              Select a session run to inspect node progress.
            </div>
          )}

          <div className="pt-4 border-t border-[#DCE4D9] mt-6">
            <button
              onClick={() => onNavigate('data-logs')}
              className="w-full py-2 bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-bold rounded-lg transition-colors text-center"
            >
              Filter Records by This Session
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
