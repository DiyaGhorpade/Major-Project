import React from 'react';
import { BrainCircuit, Cpu, PieChart, Target, Sparkles, ArrowRight, Layers } from 'lucide-react';
import { MetricCard } from '../components/shared/MetricCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { EmptyState, RunEmptyState } from '../components/shared/EmptyState';
import { ErrorState } from '../components/shared/ErrorState';
import { CountBars } from '../components/ai/CountBars';
import { orderedCounts } from '../components/ai/counts';
import { NavRoute } from '../components/layout/Sidebar';
import { useHealth } from '../hooks/useHealth';
import { useInsights, useModelMetrics } from '../hooks/useData';
import { formatDateTime } from '../lib/utils';
import type { AiInsights } from '../types';

interface AiInsightsPageProps {
  onNavigate: (route: NavRoute, plantId?: string) => void;
}

const SEVERITY_ORDER = ['healthy', 'low', 'moderate', 'high', 'unmeasured'] as const;
const SEVERITY_BAR: Record<string, string> = {
  healthy: 'bg-[#2E8B57]',
  low: 'bg-[#8DB580]',
  moderate: 'bg-[#C98227]',
  high: 'bg-[#C94C4C]',
};
const CONDITION_ORDER = [
  'Healthy',
  'Water Stress',
  'Light Stress',
  'Temperature/Humidity Stress',
  'Mixed Stress',
  'Unclassified Stress',
] as const;

const sum = (counts: Record<string, number>) => Object.values(counts).reduce((n, v) => n + v, 0);
const yesNo = (v: boolean | null | undefined) => (v === null || v === undefined ? 'n/a' : v ? 'Available' : 'Not available');

const SeverityBlock: React.FC<{ title: string; subtitle: string; counts: Record<string, number> }> = ({ title, subtitle, counts }) => {
  const entries = orderedCounts(counts, SEVERITY_ORDER);
  const total = sum(counts);
  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
      <div className="flex items-center justify-between mb-4 gap-3">
        <div>
          <h3 className="text-sm font-bold text-[#1C2B23]">{title}</h3>
          <p className="text-xs text-[#6B786E]">{subtitle}</p>
        </div>
        <span className="text-xs font-mono text-stone-500">{total.toLocaleString()} total</span>
      </div>
      <CountBars
        className="lg:grid-cols-2"
        total={total}
        items={entries.map(([key, count]) => ({ key, label: key.charAt(0).toUpperCase() + key.slice(1), count, barClassName: SEVERITY_BAR[key] }))}
      />
    </div>
  );
};

const ModelStatusCard: React.FC<{ insights: AiInsights | null }> = ({ insights }) => {
  const { health, error } = useHealth();
  const connected = health?.mlService.ok === true;
  const known = health !== null;
  const version = health?.modelVersion ?? insights?.model.modelVersion ?? null;
  const runId = health?.runId ?? insights?.model.runId ?? null;
  const explainability = insights?.model.explainabilityAvailable ?? health?.mlService.explainabilityAvailable ?? null;
  const last = insights?.lastPipelineRun ?? null;
  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card" data-testid="model-status-card">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-[#2E8B57]" aria-hidden="true" />
          <h3 className="text-sm font-bold text-[#1C2B23]">Model status</h3>
        </div>
        {known ? (
          <StatusBadge status={connected ? 'online' : 'offline'} label={connected ? 'AI model connected' : 'AI model unreachable'} size="sm" />
        ) : (
          <StatusBadge status="unknown" label={error ? 'Backend not reachable' : 'Checking AI model...'} size="sm" />
        )}
      </div>
      <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-3 text-xs">
        <div>
          <dt className="text-[#6B786E]">Model version</dt>
          <dd className="font-mono font-semibold text-[#1C2B23]">{version ? `v${version}` : 'n/a'}</dd>
        </div>
        <div>
          <dt className="text-[#6B786E]">Model run id</dt>
          <dd className="font-mono font-semibold text-[#1C2B23] break-all">{runId ?? 'n/a'}</dd>
        </div>
        <div>
          <dt className="text-[#6B786E]">Trained at (IST)</dt>
          <dd className="font-mono font-semibold text-[#1C2B23]">{formatDateTime(insights?.model.trainedAt)}</dd>
        </div>
        <div>
          <dt className="text-[#6B786E]">Explainability (SHAP)</dt>
          <dd className="font-semibold text-[#1C2B23]">{yesNo(explainability)}</dd>
        </div>
        <div className="sm:col-span-2 lg:col-span-4">
          <dt className="text-[#6B786E]">Last analysis run</dt>
          <dd className="font-semibold text-[#1C2B23]">
            {last ? (
              <>
                #{last.runId} {last.status}: {last.processed.toLocaleString()} of {last.total.toLocaleString()} readings predicted
                {last.failed > 0 ? `, ${last.failed.toLocaleString()} failed` : ''}
                {last.finishedAt ? ` (finished ${formatDateTime(last.finishedAt)} IST)` : ''}
                {last.message ? ` - ${last.message}` : ''}
              </>
            ) : (
              'No analysis run yet'
            )}
          </dd>
        </div>
      </dl>
    </div>
  );
};

export const AiInsightsPage: React.FC<AiInsightsPageProps> = ({ onNavigate }) => {
  const { health, mlConnected } = useHealth();
  const insightsQuery = useInsights();
  const metricsQuery = useModelMetrics();
  const insights = insightsQuery.data;
  const metrics = metricsQuery.data;

  const latestSeverity = insights?.latestSeverityCounts ?? {};
  const analyzedPlants = insights ? sum(Object.fromEntries(Object.entries(latestSeverity).filter(([k]) => k !== 'unmeasured'))) : 0;
  const stressedPlants = (latestSeverity.moderate ?? 0) + (latestSeverity.high ?? 0);

  let body: React.ReactNode = null;
  if (insightsQuery.error && !insights) {
    body = <ErrorState message={insightsQuery.error.message} onRetry={() => void insightsQuery.refetch()} />;
  } else if (insightsQuery.loading || !insights) {
    body = <EmptyState icon={<BrainCircuit className="w-8 h-8 text-stone-400" />} title="Loading AI insights..." description="" />;
  } else if (!insights.hasPredictions) {
    body = !mlConnected && health ? (
      <EmptyState
        icon={<Cpu className="w-8 h-8 text-stone-400" />}
        title="AI model unreachable"
        description={`The ML service is not reachable at ${health.mlService.url}. Sensor data is still available; predictions need the ML service.`}
      />
    ) : (
      <RunEmptyState />
    );
  } else {
    const totalPlants = insights.explanationCoverage.totalPlants;
    const okPredictions = insights.predictions.ok ?? 0;
    const globalFeatures = insights.topFeatures.global;
    const maxGlobal = Math.max(0, ...(globalFeatures ?? []).map((f) => f.meanAbsShap));
    const maxLocal = Math.max(0, ...insights.topFeatures.localFrequency.map((f) => f.count));
    body = (
      <>
        {insightsQuery.error && <ErrorState message={insightsQuery.error.message} onRetry={() => void insightsQuery.refetch()} />}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Predictions"
            value={okPredictions.toLocaleString()}
            subtitle={`${(insights.predictions.unavailable ?? 0).toLocaleString()} unavailable, ${(insights.predictions.rejected ?? 0).toLocaleString()} rejected`}
            icon={<BrainCircuit className="w-4 h-4 text-[#2E8B57]" />}
          />
          <MetricCard
            title="Anomalous Readings"
            value={insights.anomalies.count.toLocaleString()}
            subtitle={`${(insights.anomalies.rate * 100).toFixed(1)}% (${insights.anomalies.rateBasis})`}
            icon={<Sparkles className="w-4 h-4 text-[#2E8B57]" />}
          />
          <MetricCard
            title="Explanation Coverage"
            value={`${insights.explanationCoverage.plantsWithLocalShap} / ${totalPlants} Plants`}
            subtitle={`${stressedPlants} of ${analyzedPlants} analyzed plants rated moderate or high`}
            icon={<PieChart className="w-4 h-4 text-[#2E8B57]" />}
            onClick={() => onNavigate('shap-analysis')}
          />
          <MetricCard
            title="Hold-out Accuracy"
            value={metrics ? `${(metrics.overall.accuracy * 100).toFixed(1)}%` : 'n/a'}
            subtitle={metrics ? `N = ${metrics.total_samples.toLocaleString()} (hold-out)` : 'Evaluation unavailable'}
            icon={<Target className="w-4 h-4 text-[#2E8B57]" />}
            onClick={() => onNavigate('model-performance')}
          />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <SeverityBlock
            title="Model-rated severity (latest reading per plant)"
            subtitle="One count per plant, from its newest prediction."
            counts={insights.latestSeverityCounts}
          />
          <SeverityBlock
            title="Model-rated severity (all predictions)"
            subtitle="Every stored prediction across all plants and sessions."
            counts={insights.severityCounts}
          />
        </div>

        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
          <div className="flex items-center justify-between mb-4 gap-3">
            <div>
              <h3 className="text-sm font-bold text-[#1C2B23]">Stress-type guess (latest reading per plant)</h3>
            </div>
            <span className="text-xs font-mono text-stone-500">{sum(insights.latestConditionGuessCounts)} plants</span>
          </div>
          <CountBars
            total={sum(insights.latestConditionGuessCounts)}
            unit="plants"
            items={orderedCounts(insights.latestConditionGuessCounts, CONDITION_ORDER).map(([label, count]) => ({ key: label, label, count }))}
          />
          <details className="mt-4 text-xs text-[#6B786E]">
            <summary className="cursor-pointer font-semibold text-[#173F32]">Across all predictions</summary>
            <div className="mt-3">
              <CountBars
                total={sum(insights.conditionGuessCounts)}
                unit="readings"
                items={orderedCounts(insights.conditionGuessCounts, CONDITION_ORDER).map(([label, count]) => ({ key: label, label, count }))}
              />
            </div>
          </details>
        </div>

        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
          <div className="flex items-center justify-between mb-4 gap-3">
            <div>
              <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-[#2E8B57]" aria-hidden="true" />
                Environmental states
              </h3>
              <p className="text-xs text-[#6B786E]">
                Groups of similar environmental readings found by clustering. The group numbers are arbitrary and do not mean healthy or unhealthy.
              </p>
            </div>
            <span className="text-xs font-mono text-stone-500">{sum(Object.fromEntries(insights.envStateCounts.map((e) => [String(e.id), e.count])))} readings</span>
          </div>
          {insights.envStateCounts.length === 0 ? (
            <p className="text-xs text-[#6B786E]">No environmental state groups available.</p>
          ) : (
            <CountBars
              unit="readings"
              items={insights.envStateCounts.map((e) => ({
                key: String(e.id),
                label: e.label,
                count: e.count,
              }))}
            />
          )}
        </div>

        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
          <div className="mb-4">
            <h3 className="text-sm font-bold text-[#1C2B23]">Top features driving the model</h3>
            <p className="text-xs text-[#6B786E]">
              {globalFeatures
                ? `Global mean |SHAP| (log-odds)${insights.topFeatures.globalSource ? `, report generated ${formatDateTime(insights.topFeatures.globalSource.generatedAt)} IST` : ''}.`
                : 'The global SHAP report has not been generated. Showing how often each feature appears in the per-plant explanations instead.'}
            </p>
          </div>
          <ul className="space-y-2 text-xs">
            {globalFeatures
              ? globalFeatures.map((f) => (
                  <li key={f.name} className="flex items-center gap-3">
                    <span className="w-64 shrink-0 font-semibold text-[#1C2B23] truncate" title={f.displayName}>
                      {f.displayName}
                    </span>
                    <div className="flex-1 bg-[#EDF4EB] h-2 rounded-full overflow-hidden">
                      <div className="bg-[#2E8B57] h-full" style={{ width: `${maxGlobal > 0 ? (f.meanAbsShap / maxGlobal) * 100 : 0}%` }} />
                    </div>
                    <span className="w-14 text-right font-mono text-[#6B786E]">{f.meanAbsShap.toFixed(3)}</span>
                  </li>
                ))
              : insights.topFeatures.localFrequency.map((f) => (
                  <li key={f.name} className="flex items-center gap-3">
                    <span className="w-64 shrink-0 font-semibold text-[#1C2B23] truncate" title={f.displayName}>
                      {f.displayName}
                    </span>
                    <div className="flex-1 bg-[#EDF4EB] h-2 rounded-full overflow-hidden">
                      <div className="bg-[#2E8B57] h-full" style={{ width: `${maxLocal > 0 ? (f.count / maxLocal) * 100 : 0}%` }} />
                    </div>
                    <span className="w-14 text-right font-mono text-[#6B786E]">{f.count} plants</span>
                  </li>
                ))}
          </ul>
        </div>

        {/* Navigation cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            type="button"
            onClick={() => onNavigate('shap-analysis')}
            className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group text-left block w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2E8B57]"
          >
            <span className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
              <PieChart className="w-5 h-5" />
            </span>
            <span className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1 block">SHAP Explainability Workspace</span>
            <span className="text-xs text-[#6B786E] leading-relaxed mb-4 block">
              Explore local waterfall attributions and global feature impact for the model's predictions.
            </span>
            <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
              Open SHAP Workspace <ArrowRight className="w-3.5 h-3.5" />
            </span>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('model-performance')}
            className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group text-left block w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2E8B57]"
          >
            <span className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
              <Target className="w-5 h-5" />
            </span>
            <span className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1 block">Model Performance & Matrix</span>
            <span className="text-xs text-[#6B786E] leading-relaxed mb-4 block">
              Review the hold-out confusion matrix, precision/recall and per-class metrics.
            </span>
            <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
              View Performance Metrics <ArrowRight className="w-3.5 h-3.5" />
            </span>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('recommendations')}
            className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group text-left block w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2E8B57]"
          >
            <span className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
              <Sparkles className="w-5 h-5" />
            </span>
            <span className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1 block">Plant Care Recommendations</span>
            <span className="text-xs text-[#6B786E] leading-relaxed mb-4 block">
              Actions built from the model's output and recent sensor values.
            </span>
            <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
              Review Care Actions <ArrowRight className="w-3.5 h-3.5" />
            </span>
          </button>
                </div>
      </>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">ML Inference Service</span>
            <h1 className="text-lg font-bold text-[#1C2B23]">Plant Stress & Microclimate ML Insights</h1>
          </div>
        </div>
        {insightsQuery.refreshing && <span className="text-xs text-[#6B786E]">Refreshing...</span>}
      </div>

      <ModelStatusCard insights={insights} />
      {body}
    </div>
  );
};
