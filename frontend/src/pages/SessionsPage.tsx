import React, { useState } from 'react';
import { History, Clock, ChevronRight, ShieldCheck, Cpu } from 'lucide-react';
import { useSessions } from '../hooks/useData';
import { useSettings } from '../hooks/useSettings';
import { StatusBadge } from '../components/shared/StatusBadge';
import { ErrorState } from '../components/shared/ErrorState';
import { usePlants } from '../hooks/usePlants';
import { describeCadence, formatDuration, observedCadence, stashSessionFilter } from '../components/monitoring/format';
import { formatDateTime, formatCount } from '../lib/utils';
import { NavRoute } from '../components/layout/Sidebar';

interface SessionsPageProps {
  onNavigate: (route: NavRoute) => void;
}

export const SessionsPage: React.FC<SessionsPageProps> = ({ onNavigate }) => {
  const sessionsQuery = useSessions();
  const settingsQuery = useSettings();
  const sessions = sessionsQuery.data ?? [];
  const [pickedId, setPickedId] = useState<number | null>(null);
  const selectedSession = sessions.find((s) => s.session_id === pickedId) ?? sessions[0] ?? null;

  const plantCount = usePlants().data?.length ?? 0;
  const times = settingsQuery.data?.collection_times;
  const configured = describeCadence(times);
  // What the stored data shows (from the latest session), shown apart from the configured schedule.
  const observed = sessions[0] ? observedCadence(sessions[0], plantCount) : null;

  return (
    <div className="space-y-6">
      {sessionsQuery.error && <ErrorState message={sessionsQuery.error.message} onRetry={() => void sessionsQuery.refetch()} />}

      {/* Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">Ingestion Session Registry</span>
            {observed && (
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]" data-testid="observed-cadence">
                Data shows readings {observed}
              </span>
            )}
            {configured && (
              <span
                className="text-xs px-2.5 py-0.5 rounded-full bg-white text-[#426B4F] font-semibold border border-[#DCE4D9]"
                data-testid="configured-schedule"
                title={times ? `Collection times (IST): ${[...times].sort().join(', ')}` : undefined}
              >
                Configured schedule: {configured}
              </span>
            )}
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">Historical Collection Runs &amp; Run Audits</h1>
          <p className="text-xs text-[#6B786E] mt-1">
            Each session groups a block of the bundled dataset&apos;s readings. Node progress compares received readings with what the schedule expected.
          </p>
        </div>

        <button
          onClick={() => {
            stashSessionFilter(null);
            onNavigate('data-logs');
          }}
          className="px-3.5 py-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-semibold flex items-center gap-2 transition-colors self-start md:self-auto"
        >
          <History className="w-3.5 h-3.5" />
          <span>View All Log Records</span>
        </button>
      </div>

      {sessionsQuery.loading && <div className="p-6 text-center text-xs text-stone-500">Loading sessions...</div>}
      {!sessionsQuery.loading && !sessionsQuery.error && sessions.length === 0 && (
        <div className="rounded-card border border-[#DCE4D9] bg-white p-8 text-center text-xs text-stone-500">No sessions recorded.</div>
      )}

      {sessions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Session list */}
          <div className="lg:col-span-2 bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
            <div className="p-4 border-b border-[#DCE4D9] flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#1C2B23]">Collection Passes</h3>
              <span className="text-xs font-mono text-stone-500">{sessions.length} sessions logged</span>
            </div>

            <div className="divide-y divide-[#DCE4D9]/60">
              {sessions.map((s) => {
                const isSelected = selectedSession?.session_id === s.session_id;
                return (
                  <div
                    key={s.session_id}
                    data-testid={`session-row-${s.session_id}`}
                    onClick={() => setPickedId(s.session_id)}
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
                          <span className="font-mono text-xs font-bold text-[#1C2B23]">Session #{s.session_id}</span>
                          <StatusBadge status={s.status} size="sm" />
                        </div>
                        <div className="text-[11px] text-[#6B786E]">
                          Started: {formatDateTime(s.started_at)} • Span: {formatDuration(s.duration_seconds)}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-right">
                      <div>
                        <div className="text-xs font-bold text-[#1C2B23]">
                          {formatCount(s.received_records)} / {formatCount(s.expected_records)}
                        </div>
                        <div className="text-[10px] font-medium">
                          <span className={s.rejected_records > 0 ? 'text-[#C94C4C] font-bold' : 'text-stone-500'}>{s.rejected_records} rejected</span>
                          {' • '}
                          <span className={s.flagged_records > 0 ? 'text-amber-700 font-bold' : 'text-stone-500'}>{s.flagged_records} flagged</span>
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-stone-400" />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Selected session */}
          <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col justify-between" data-testid="session-detail">
            {selectedSession && (
              <div className="space-y-5">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-[#426B4F]">Session Audit</span>
                    <StatusBadge status={selectedSession.status} size="sm" />
                  </div>
                  <h3 className="text-base font-bold text-[#1C2B23] font-mono">Session #{selectedSession.session_id}</h3>
                  <p className="text-[11px] text-stone-500">Trigger: {selectedSession.trigger_type}</p>
                </div>

                <div className="p-3.5 rounded-xl bg-stone-50 border border-stone-200 text-xs space-y-2">
                  <div className="flex justify-between">
                    <span className="text-[#6B786E]">Scheduled Time:</span>
                    <span className="font-mono text-[#1C2B23]">{formatDateTime(selectedSession.scheduled_at)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B786E]">Ended:</span>
                    <span className="font-mono text-[#1C2B23]">{selectedSession.ended_at ? formatDateTime(selectedSession.ended_at) : 'In Progress'}</span>
                  </div>
                  <div className="flex justify-between font-semibold border-t border-stone-200 pt-2">
                    <span className="text-[#6B786E]">Time Span:</span>
                    <span className="text-[#2E8B57]">{formatDuration(selectedSession.duration_seconds)}</span>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-[#2E8B57]" /> Node Progress
                  </h4>
                  <div className="space-y-2.5">
                    {Object.entries(selectedSession.node_progress).map(([nodeId, prog]) => (
                      <div key={nodeId} className="p-3 rounded-lg border border-[#DCE4D9] bg-white text-xs" data-testid={`node-progress-${nodeId}`}>
                        <div className="flex justify-between items-center mb-1 font-semibold text-[#1C2B23]">
                          <span className="flex items-center gap-2">
                            Node {nodeId} <StatusBadge status={prog.status} size="sm" />
                          </span>
                          <span className="text-[#2E8B57]">
                            {formatCount(prog.received)} / {formatCount(prog.expected)} readings
                          </span>
                        </div>
                        <div className="w-full bg-[#EDF4EB] h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-[#2E8B57] h-full"
                            style={{ width: `${prog.expected > 0 ? Math.min(100, (prog.received / prog.expected) * 100) : 0}%` }}
                          />
                        </div>
                      </div>
                    ))}
                    {Object.keys(selectedSession.node_progress).length === 0 && <p className="text-xs text-stone-500">No node progress recorded.</p>}
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-start gap-2">
                  <ShieldCheck className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                  <span>
                    <strong data-testid="flagged-count">{formatCount(selectedSession.flagged_records)}</strong> flagged and{' '}
                    {formatCount(selectedSession.rejected_records)} rejected of {formatCount(selectedSession.received_records)} received readings
                    {selectedSession.flagged_records > 0 ? ' (flagged: missing or out-of-range values, kept as stored; see Data Logs).' : '.'}
                  </span>
                </div>
              </div>
            )}

            {selectedSession && (
              <div className="pt-4 border-t border-[#DCE4D9] mt-6">
                <button
                  onClick={() => {
                    stashSessionFilter(selectedSession.session_id);
                    onNavigate('data-logs');
                  }}
                  className="w-full py-2 bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-bold rounded-lg transition-colors text-center"
                >
                  Filter Records by This Session
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
