import React, { useEffect, useMemo, useState } from 'react';
import { AlertCircle, ChevronLeft, ChevronRight, Search, X } from 'lucide-react';
import { useRecommendations } from '../hooks/useRecommendations';
import { useRecommendationStatus } from '../hooks/useRecommendationStatus';
import { usePlants } from '../hooks/usePlants';
import { ErrorState } from '../components/shared/ErrorState';
import { RecommendationCard } from '../components/shared/RecommendationCard';
import { EmptyState, RunEmptyState } from '../components/shared/EmptyState';
import { useRecommendationCounts } from '../components/ai/useRecommendationCounts';
import type { RecommendationFilter } from '../api/client';
import { PAGE_SIZE, describeRecommendationError } from '../components/ai/recommendationHelpers';
import type {
  RecommendationCategory,
  RecommendationPriority,
  RecommendationSource,
  RecommendationStatus,
  UserRecommendationStatus,
} from '../types';
import { NavRoute } from '../components/layout/Sidebar';

interface RecommendationsPageProps {
  onNavigate: (route: NavRoute, plantId?: number | string) => void;
}

const SEARCH_DEBOUNCE_MS = 300;

const SELECT_CLASS =
  'text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer';

export const RecommendationsPage: React.FC<RecommendationsPageProps> = ({ onNavigate }) => {
  const [status, setStatus] = useState<RecommendationStatus | 'all'>('all');
  const [priority, setPriority] = useState<RecommendationPriority | 'all'>('all');
  const [category, setCategory] = useState<RecommendationCategory | 'all'>('all');
  const [source, setSource] = useState<RecommendationSource | 'all'>('all');
  const [plantId, setPlantId] = useState<string>('all');
  const [includeSuperseded, setIncludeSuperseded] = useState(false);
  const [searchInput, setSearchInput] = useState(''); // what the user typed
  const [searchQuery, setSearchQuery] = useState(''); // what is sent to the server (debounced)
  const [pageIdx, setPageIdx] = useState(0);
  // Optimistic status changes by recommendation id. An override only applies while the server still reports
  // the status it was made from, so it disappears by itself once the refetched list agrees (or on rollback).
  const [overrides, setOverrides] = useState<Record<string, { from: RecommendationStatus; to: RecommendationStatus }>>({});

  const plants = usePlants().data ?? [];

  const filter: RecommendationFilter = useMemo(
    () => ({
      status: status === 'all' ? undefined : status,
      priority: priority === 'all' ? undefined : priority,
      category: category === 'all' ? undefined : category,
      source: source === 'all' ? undefined : source,
      plantId: plantId === 'all' ? undefined : Number(plantId),
      q: searchQuery || undefined,
      includeSuperseded: includeSuperseded || undefined,
      limit: PAGE_SIZE,
      offset: pageIdx * PAGE_SIZE,
    }),
    [status, priority, category, source, plantId, searchQuery, includeSuperseded, pageIdx],
  );
  const recsQuery = useRecommendations(filter);
  const statusChange = useRecommendationStatus();
  const countsQuery = useRecommendationCounts();
  const counts = countsQuery.data;
  const recommendations = recsQuery.data;
  const total = recsQuery.total;
  const lastPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

  // Search runs on the server (all pages, counted in the total). Typing is debounced; a new search starts at page 1.
  useEffect(() => {
    const typed = searchInput.trim();
    if (typed === searchQuery) return;
    const id = setTimeout(() => {
      setSearchQuery(typed);
      setPageIdx(0);
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [searchInput, searchQuery]);

  // Clamp the page when the list shrinks (adjusting state during render, not in an effect).
  if (recommendations && pageIdx > lastPage) setPageIdx(lastPage);

  const changeFilter = <T,>(setter: (v: T) => void) => (value: T) => {
    setter(value);
    setPageIdx(0);
    statusChange.clearError();
  };

  const handleUpdateStatus = async (id: string, next: UserRecommendationStatus) => {
    const current = (recommendations ?? []).find((r) => r.id === id);
    if (!current || current.status === next) return;
    setOverrides((o) => ({ ...o, [id]: { from: current.status, to: next } }));
    // Roll back the optimistic change only if this is still the latest request for the id; the error banner explains why.
    await statusChange.update(id, next, () =>
      setOverrides((o) => {
        const rest = { ...o };
        delete rest[id];
        return rest;
      }),
    );
  };

  const shown = useMemo(
    () =>
      (recommendations ?? []).map((r) => {
        const o = overrides[r.id];
        return o && o.from === r.status ? { ...r, status: o.to } : r;
      }),
    [recommendations, overrides],
  );

  const filtersActive =
    status !== 'all' || priority !== 'all' || category !== 'all' || source !== 'all' || plantId !== 'all' || includeSuperseded;
  const from = total === 0 ? 0 : pageIdx * PAGE_SIZE + 1;
  const to = Math.min(total, (pageIdx + 1) * PAGE_SIZE);
  const mutationMessage = statusChange.error ? describeRecommendationError(statusChange.error, recommendations ?? []) : null;

  const countChips: Array<{ key: RecommendationStatus; label: string; className: string; valueClass: string }> = [
    { key: 'new', label: 'New Actions', className: 'bg-amber-50 border-amber-200 text-amber-900', valueClass: 'text-amber-900' },
    { key: 'in_progress', label: 'In Progress', className: 'bg-blue-50 border-blue-200 text-blue-900', valueClass: 'text-blue-900' },
    { key: 'completed', label: 'Completed', className: 'bg-[#EDF4EB] border-[#DDE8D8] text-[#173F32]', valueClass: 'text-[#2E8B57]' },
    { key: 'dismissed', label: 'Dismissed', className: 'bg-stone-50 border-stone-200 text-stone-800', valueClass: 'text-stone-700' },
    { key: 'superseded', label: 'Superseded', className: 'bg-stone-50 border-stone-200 text-stone-600', valueClass: 'text-stone-500' },
  ];

  return (
    <div className="space-y-6">
      {/* Header & Status Summary */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col xl:flex-row xl:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">Agronomic Decision Support</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              ML-driven
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">Plant Care Recommendations</h1>
          <p className="text-xs text-[#6B786E] mt-1 max-w-2xl leading-relaxed">
            Actions built from the ML model's output and recent sensor values. Cards show the sensor values behind each action, where available.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs" data-testid="status-counts">
          {countChips.map((chip) => (
            <div key={chip.key} className={`px-3 py-2 rounded-xl border font-semibold text-center min-w-[78px] ${chip.className}`}>
              <span className={`block text-sm font-bold ${chip.valueClass}`} data-testid={`count-${chip.key}`}>
                {counts ? counts[chip.key].toLocaleString() : '-'}
              </span>
              <span className="text-[10px]">{chip.label}</span>
            </div>
          ))}
        </div>
      </div>

      {recsQuery.error && !recommendations && (
        <ErrorState title="Could not load recommendations" message={recsQuery.error.message} onRetry={() => void recsQuery.refetch()} />
      )}
      {mutationMessage && (
        <div
          role="alert"
          data-testid="mutation-error"
          className="p-3.5 rounded-xl bg-red-50 border border-red-200 text-xs text-red-900 flex items-start gap-3"
        >
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="flex-1 leading-relaxed">
            <strong className="font-semibold">Status not changed. </strong>
            {mutationMessage}
          </div>
          <button type="button" onClick={statusChange.clearError} aria-label="Dismiss message" className="text-red-700 hover:text-red-900">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 flex-1">
          <div className="relative min-w-[200px] flex-1 max-w-xs">
            <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search title, action, reason or plant..."
              aria-label="Search recommendations"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-full text-xs bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg pl-9 pr-3 py-1.5 outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
            />
          </div>

          <select aria-label="Status filter" value={status} onChange={(e) => changeFilter(setStatus)(e.target.value as RecommendationStatus | 'all')} className={SELECT_CLASS}>
            <option value="all">All Statuses</option>
            <option value="new">New</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
            <option value="dismissed">Dismissed</option>
            <option value="superseded">Superseded</option>
          </select>

          <select aria-label="Priority filter" value={priority} onChange={(e) => changeFilter(setPriority)(e.target.value as RecommendationPriority | 'all')} className={SELECT_CLASS}>
            <option value="all">All Priorities</option>
            <option value="high">High Priority</option>
            <option value="medium">Medium Priority</option>
            <option value="low">Low Priority</option>
          </select>

          <select aria-label="Category filter" value={category} onChange={(e) => changeFilter(setCategory)(e.target.value as RecommendationCategory | 'all')} className={SELECT_CLASS}>
            <option value="all">All Categories</option>
            <option value="irrigation">Watering</option>
            <option value="lighting">Lighting</option>
            <option value="climate">Climate</option>
            <option value="sensor_check">Sensor check</option>
            <option value="monitoring">Monitoring</option>
          </select>

          <select aria-label="Source filter" value={source} onChange={(e) => changeFilter(setSource)(e.target.value as RecommendationSource | 'all')} className={SELECT_CLASS}>
            <option value="all">All Sources</option>
            <option value="ml">ML</option>
            <option value="heuristic">Backend rule</option>
          </select>

          <select aria-label="Plant filter" value={plantId} onChange={(e) => changeFilter(setPlantId)(e.target.value)} className={SELECT_CLASS}>
            <option value="all">All Plants</option>
            {plants.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          <label className="inline-flex items-center gap-1.5 text-xs text-[#1C2B23] cursor-pointer">
            <input
              type="checkbox"
              checked={includeSuperseded}
              onChange={(e) => changeFilter(setIncludeSuperseded)(e.target.checked)}
              className="accent-[#2E8B57]"
            />
            Show superseded
          </label>
        </div>

        <span className="text-xs font-mono text-stone-500" data-testid="range-label">
          {total === 0 ? '0 directives' : `Showing ${from}-${to} of ${total.toLocaleString()} directives`}
          {recsQuery.refreshing ? ' (refreshing)' : ''}
        </span>
      </div>

      {/* Cards */}
      {recsQuery.loading ? (
        <EmptyState title="Loading recommendations..." description="" />
      ) : shown.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {shown.map((rec) => (
            <RecommendationCard
              key={rec.id}
              recommendation={rec}
              pending={statusChange.isPending(rec.id)}
              onUpdateStatus={(id, next) => void handleUpdateStatus(id, next)}
              onViewPlant={(id) => onNavigate('plant-detail', id)}
            />
          ))}
        </div>
      ) : recommendations && !filtersActive && total === 0 && !searchQuery && !searchInput.trim() ? (
        <RunEmptyState title="No recommendations yet" />
      ) : recommendations ? (
        <EmptyState
          title="No Recommendations Match"
          description={
            searchQuery
              ? 'No recommendations match the search text and filters. Clear the search or adjust the filters.'
              : 'No recommendations match the selected filters. Try broadening them, or show superseded recommendations.'
          }
        />
      ) : null}

      {/* Pagination */}
      {total > PAGE_SIZE && (
        <div className="flex items-center justify-center gap-3 text-xs" data-testid="pagination">
          <button
            type="button"
            onClick={() => setPageIdx((p) => Math.max(0, p - 1))}
            disabled={pageIdx === 0}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-[#DCE4D9] bg-white disabled:opacity-50 hover:border-[#2E8B57]"
          >
            <ChevronLeft className="w-3.5 h-3.5" /> Previous
          </button>
          <span className="font-mono text-stone-600">
            Page {pageIdx + 1} of {lastPage + 1}
          </span>
          <button
            type="button"
            onClick={() => setPageIdx((p) => Math.min(lastPage, p + 1))}
            disabled={pageIdx >= lastPage}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-[#DCE4D9] bg-white disabled:opacity-50 hover:border-[#2E8B57]"
          >
            Next <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};
