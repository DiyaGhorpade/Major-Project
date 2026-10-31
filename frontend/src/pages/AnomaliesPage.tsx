import React, { useState } from 'react';
import { Activity, ChevronLeft, ChevronRight, Clock, ScanSearch, Trees } from 'lucide-react';
import type { AnomalyFilter } from '../api/client';
import { MetricCard } from '../components/shared/MetricCard';
import { EmptyState, RunEmptyState } from '../components/shared/EmptyState';
import { ErrorState } from '../components/shared/ErrorState';
import { NavRoute } from '../components/layout/Sidebar';
import { useAnomalies, useAnomalySummary } from '../hooks/useData';
import { usePlants } from '../hooks/usePlants';
import { MIN_SCORE_MAX, MIN_SCORE_MIN, parseAnomalyPlantId, parseMinScore } from '../lib/anomalyFilters';
import { formatCount, formatDateTime, formatFraction, formatLux, formatPercent, formatTemp } from '../lib/utils';

interface AnomaliesPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
  /** Pre-selects the plant filter (e.g. "View all anomalies for this plant" on the plant page). */
  initialPlantId?: number | string | null;
}

type SortKey = 'recent' | 'score';

const SELECT_CLASS = 'text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer';
const fmtScore = (v: number) => v.toFixed(3);

/** A small bar (score as a share of the highest score seen) with the number, plus the threshold in the tooltip. */
const ScoreMeter: React.FC<{ score: number; threshold: number; max: number }> = ({ score, threshold, max }) => {
  const pct = max > 0 ? Math.min(100, Math.max(0, (score / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-2" title={`score ${fmtScore(score)} vs threshold ${fmtScore(threshold)}`}>
      <div className="w-20 bg-[#EDF4EB] h-1.5 rounded-full overflow-hidden" aria-hidden="true">
        <div className="bg-[#C98227] h-full" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-bold text-[#1C2B23]">{fmtScore(score)}</span>
      <span className="text-[10px] text-stone-500 font-sans">vs {fmtScore(threshold)}</span>
    </div>
  );
};

export const AnomaliesPage: React.FC<AnomaliesPageProps> = ({ onNavigate, initialPlantId = null }) => {
  const [plantFilter, setPlantFilter] = useState<string>(parseAnomalyPlantId(initialPlantId) ?? 'all');
  const [sort, setSort] = useState<SortKey>('recent');
  const [minScoreText, setMinScoreText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(15);

  const minScore = parseMinScore(minScoreText);
  const filter: AnomalyFilter = {
    plantId: plantFilter,
    sort,
    minScore,
    limit: pageSize,
    offset: (currentPage - 1) * pageSize,
  };

  const summaryQuery = useAnomalySummary();
  const list = useAnomalies(filter);
  const plantsQuery = usePlants();
  const summary = summaryQuery.data;
  const plants = plantsQuery.data ?? [];
  const rows = list.data ?? [];
  const total = list.total;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  // The total can shrink under the current page (filter change, new run): step back to the last page that exists.
  if (list.data !== null && currentPage > totalPages) setCurrentPage(totalPages);
  const startIndex = (currentPage - 1) * pageSize;
  const maxScore = Math.max(0, ...(summary?.plants ?? []).map((p) => p.maxScore), ...rows.map((r) => r.score));
  const filtersActive = plantFilter !== 'all' || minScore !== undefined;

  const changeFilter = <T,>(set: (v: T) => void) => (value: T) => {
    set(value);
    setCurrentPage(1);
  };

  const header = (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center">
          <ScanSearch className="w-5 h-5" />
        </div>
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">Isolation Forest</span>
          <h1 className="text-lg font-bold text-[#1C2B23]">Unusual Readings</h1>
          <p className="text-xs text-[#6B786E] mt-0.5">
            Readings whose combination of soil, temperature, humidity and light looked unusual compared with typical readings.
          </p>
        </div>
      </div>
      {(summaryQuery.refreshing || list.refreshing) && <span className="text-xs text-[#6B786E]">Refreshing...</span>}
    </div>
  );

  if (summaryQuery.error && !summary) {
    return (
      <div className="space-y-6">
        {header}
        <ErrorState message={summaryQuery.error.message} onRetry={() => void summaryQuery.refetch()} />
      </div>
    );
  }
  if (summaryQuery.loading || !summary) {
    return (
      <div className="space-y-6">
        {header}
        <EmptyState icon={<ScanSearch className="w-8 h-8 text-stone-400" />} title="Loading unusual readings..." description="" />
      </div>
    );
  }
  if (summary.totalScored === 0) {
    return (
      <div className="space-y-6">
        {header}
        <RunEmptyState />
      </div>
    );
  }
  if (summary.flagged === 0) {
    return (
      <div className="space-y-6">
        {header}
        <EmptyState
          icon={<ScanSearch className="w-8 h-8 text-stone-400" />}
          title="No unusual readings"
          description={`None of the ${formatCount(summary.totalScored)} analyzed readings scored above the threshold.`}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {header}

      {summaryQuery.error && <ErrorState message={summaryQuery.error.message} onRetry={() => void summaryQuery.refetch()} />}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Flagged Readings"
          value={formatCount(summary.flagged)}
          subtitle={`${formatFraction(summary.rate)} of ${formatCount(summary.totalScored)} analyzed readings`}
          icon={<Activity className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Plants Affected"
          value={summary.plantsAffected}
          subtitle="plants with at least one unusual reading"
          icon={<Trees className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Latest Unusual Reading"
          value={formatDateTime(summary.latestFlaggedAt)}
          subtitle="IST, time the reading was taken"
          icon={<Clock className="w-4 h-4 text-[#2E8B57]" />}
        />
      </div>

      {/* Affected plants */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden" data-testid="affected-plants">
        <div className="p-5 border-b border-[#DCE4D9]">
          <h3 className="text-sm font-bold text-[#1C2B23]">Affected plants</h3>
          <p className="text-xs text-[#6B786E]">Plants with at least one unusual reading, most flags first.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">Plant</th>
                <th className="py-3 px-4 font-semibold">Group</th>
                <th className="py-3 px-4 font-semibold">Flags</th>
                <th className="py-3 px-4 font-semibold">Max score</th>
                <th className="py-3 px-4 font-semibold">Last flagged (IST)</th>
                <th className="py-3 px-4 font-semibold text-right">Readings</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60">
              {summary.plants.map((p) => (
                <tr key={p.plantId} className="hover:bg-[#EDF4EB]/40 transition-colors">
                  <td className="py-3 px-4">
                    <button
                      type="button"
                      onClick={() => onNavigate('plant-detail', p.plantId)}
                      className="font-bold text-[#2E8B57] hover:underline text-left"
                    >
                      {p.plantName}
                    </button>
                  </td>
                  <td className="py-3 px-4 text-stone-600" title={p.groupLabel}>
                    {p.groupLabel}
                  </td>
                  <td className="py-3 px-4 font-mono font-bold text-[#1C2B23]">{formatCount(p.flags)}</td>
                  <td className="py-3 px-4 font-mono text-[#1C2B23]">{fmtScore(p.maxScore)}</td>
                  <td className="py-3 px-4 font-mono text-stone-600" title={p.lastFlaggedAt ?? undefined}>
                    {formatDateTime(p.lastFlaggedAt)}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button
                      type="button"
                      onClick={() => {
                        setPlantFilter(String(p.plantId));
                        setCurrentPage(1);
                      }}
                      className="px-2.5 py-1 text-xs rounded bg-[#EDF4EB] text-[#2E8B57] font-semibold hover:bg-[#DDE8D8] transition-colors"
                    >
                      Show readings
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Filter toolbar */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 flex-1">
          <select aria-label="Plant filter" value={plantFilter} onChange={(e) => changeFilter(setPlantFilter)(e.target.value)} className={SELECT_CLASS}>
            <option value="all">All Plants ({plants.length || 'n/a'})</option>
            {plants.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          <select aria-label="Sort" value={sort} onChange={(e) => changeFilter(setSort)(e.target.value as SortKey)} className={SELECT_CLASS}>
            <option value="recent">Most recent first</option>
            <option value="score">Highest score first</option>
          </select>

          <label className="flex items-center gap-2 text-xs text-[#6B786E]">
            <span>Min score</span>
            <input
              type="number"
              step="0.01"
              min={MIN_SCORE_MIN}
              max={MIN_SCORE_MAX}
              inputMode="decimal"
              aria-label="Minimum score"
              placeholder="none"
              value={minScoreText}
              onChange={(e) => changeFilter(setMinScoreText)(e.target.value)}
              className="w-24 text-xs bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg px-3 py-1.5 outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
            />
          </label>

          {filtersActive && (
            <button
              type="button"
              onClick={() => {
                setPlantFilter('all');
                setMinScoreText('');
                setCurrentPage(1);
              }}
              className="text-xs font-semibold text-[#2E8B57] hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>

        <div className="flex items-center gap-2 text-xs text-[#6B786E]">
          <span>Per page:</span>
          <select
            aria-label="Per page"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="bg-[#EDF4EB] border border-[#DDE8D8] rounded px-2 py-1 font-semibold text-[#173F32] outline-none cursor-pointer"
          >
            <option value={10}>10</option>
            <option value={15}>15</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
          </select>
        </div>
      </div>

      {list.error && <ErrorState message={list.error.message} onRetry={() => void list.refetch()} />}

      {/* Flagged readings */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden" data-testid="anomaly-readings">
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">Time (IST)</th>
                <th className="py-3 px-4 font-semibold">Plant</th>
                <th className="py-3 px-4 font-semibold">Score vs threshold</th>
                <th className="py-3 px-4 font-semibold">Soil</th>
                <th className="py-3 px-4 font-semibold">Temperature</th>
                <th className="py-3 px-4 font-semibold">Humidity</th>
                <th className="py-3 px-4 font-semibold">Light</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60 font-mono">
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-8 px-4 text-center text-stone-500 font-sans">
                    {list.loading
                      ? 'Loading readings...'
                      : filtersActive
                        ? 'No unusual readings match these filters.'
                        : 'No unusual readings.'}
                  </td>
                </tr>
              )}
              {rows.map((a) => (
                <tr key={a.id} className="hover:bg-[#EDF4EB]/40 transition-colors">
                  <td className="py-3 px-4 text-[#1C2B23] font-medium" title={a.timestamp}>
                    {formatDateTime(a.timestamp)}
                  </td>
                  <td className="py-3 px-4 font-sans">
                    <button
                      type="button"
                      onClick={() => onNavigate('plant-detail', a.plant_id)}
                      className="font-bold text-[#2E8B57] hover:underline text-left"
                    >
                      {a.plant_name}
                    </button>
                  </td>
                  <td className="py-3 px-4">
                    <ScoreMeter score={a.score} threshold={a.threshold} max={maxScore} />
                  </td>
                  <td className="py-3 px-4 font-bold text-[#2E8B57]">{formatPercent(a.readings.soil)}</td>
                  <td className="py-3 px-4 text-[#1C2B23] font-semibold">{formatTemp(a.readings.temperature)}</td>
                  <td className="py-3 px-4 text-[#1C2B23]">{formatPercent(a.readings.humidity)}</td>
                  <td className="py-3 px-4 text-[#1C2B23]">{formatLux(a.readings.light)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="p-4 border-t border-[#DCE4D9] bg-[#F6F7F0]/40 flex flex-wrap items-center justify-between gap-3 text-xs text-[#6B786E]">
          <div>
            Showing <span className="font-semibold text-[#1C2B23]">{total === 0 ? 0 : startIndex + 1}</span> to{' '}
            <span className="font-semibold text-[#1C2B23]">{Math.min(startIndex + pageSize, total)}</span> of{' '}
            <span className="font-semibold text-[#1C2B23]">{formatCount(total)}</span> readings
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              aria-label="Previous page"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-md border border-[#DCE4D9] bg-white text-[#1C2B23] disabled:opacity-40 hover:bg-[#EDF4EB] transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-3 font-semibold text-[#1C2B23]">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              aria-label="Next page"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-md border border-[#DCE4D9] bg-white text-[#1C2B23] disabled:opacity-40 hover:bg-[#EDF4EB] transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
