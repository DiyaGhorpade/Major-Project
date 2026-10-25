import React from 'react';
import {
  BrainCircuit,
  Cpu,
  PieChart,
  Target,
  Sparkles,
  ArrowRight,
} from 'lucide-react';
import { MetricCard } from '../components/shared/MetricCard';
import { StatusBadge } from '../components/shared/StatusBadge';
import { EmptyState, RunEmptyState } from '../components/shared/EmptyState';
import { ErrorState } from '../components/shared/ErrorState';
import { AnomalyTip } from '../components/shared/Disclaimers';
import { NavRoute } from '../components/layout/Sidebar';
import { useHealth } from '../hooks/useHealth';
import { useInsights, useModelMetrics } from '../hooks/useData';

interface AiInsightsPageProps {
  onNavigate: (route: NavRoute, plantId?: string) => void;
}

export const AiInsightsPage: React.FC<AiInsightsPageProps> = ({ onNavigate }) => {
  const { health, mlConnected } = useHealth();
  const insightsQuery = useInsights();
  const metricsQuery = useModelMetrics();
  const insights = insightsQuery.data;
  const metrics = metricsQuery.data;
  const isModelConnected = mlConnected;

  // Latest heuristic stress-type guess per plant (a guess, not a diagnosis).
  const conditionCounts: Record<string, number> = insights?.latestConditionGuessCounts ?? {};
  const latestSeverity = insights?.latestSeverityCounts ?? {};
  const attentionFlags = (latestSeverity.moderate ?? 0) + (latestSeverity.high ?? 0);
  const diagnosed = insights?.explanationCoverage.totalPlants
    ? Object.entries(latestSeverity).reduce((n, [k, v]) => (k === 'unmeasured' ? n : n + v), 0)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header & Connection Toggle */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
                ML Inference Service
              </span>
              <StatusBadge status={isModelConnected ? 'online' : 'offline'} label={isModelConnected ? 'AI model connected' : 'AI model unreachable'} size="sm" />
            </div>
            <h1 className="text-lg font-bold text-[#1C2B23]">
              Plant Stress & Microclimate ML Insights
            </h1>
          </div>
        </div>

      </div>

      {insightsQuery.error && (
        <ErrorState message={insightsQuery.error.message} onRetry={() => void insightsQuery.refetch()} />
      )}

      {!isModelConnected && !insights?.hasPredictions ? (
        <EmptyState
          icon={<Cpu className="w-8 h-8 text-stone-400" />}
          title="AI model unreachable"
          description={
            health
              ? `The ML service is not reachable at ${health.mlService.url}. Sensor data is still available; predictions need the ML service.`
              : 'Checking the ML service...'
          }
        />
      ) : insights && !insights.hasPredictions ? (
        <RunEmptyState />
      ) : !insights ? null : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Classifier Model Version"
              value={insights.model.modelVersion ? `v${insights.model.modelVersion}` : 'n/a'}
              subtitle={insights.model.runId ?? 'XGBoost classifier'}
              icon={<Cpu className="w-4 h-4 text-[#2E8B57]" />}
              badge={<StatusBadge status={isModelConnected ? 'online' : 'offline'} size="sm" />}
            />
            <MetricCard
              title="Anomalous Readings"
              value={insights.anomalies.count.toLocaleString()}
              subtitle={insights.anomalies.interpretation}
              icon={
                <span className="inline-flex items-center gap-1">
                  <Sparkles className="w-4 h-4 text-[#2E8B57]" />
                  <AnomalyTip />
                </span>
              }
            />
            <MetricCard
              title="Hold-out Accuracy"
              value={metrics ? `${(metrics.overall.accuracy * 100).toFixed(1)}%` : 'n/a'}
              subtitle={metrics ? `N = ${metrics.total_samples.toLocaleString()} (synthetic labels)` : 'Evaluation unavailable'}
              icon={<Target className="w-4 h-4 text-[#2E8B57]" />}
              onClick={() => onNavigate('model-performance')}
            />
            <MetricCard
              title="Analyzed Plants"
              value={`${diagnosed} / ${insights.explanationCoverage.totalPlants} Plants`}
              subtitle={`${attentionFlags} plants rated moderate or high`}
              icon={<BrainCircuit className="w-4 h-4 text-[#2E8B57]" />}
            />
          </div>

          {/* Condition Distribution Overview */}
          <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-bold text-[#1C2B23]">Heuristic Stress-Type Guess (latest reading per plant)</h3>
                <p className="text-xs text-[#6B786E]">A heuristic guess from the model's recommendation categories, not a diagnosis.</p>
              </div>
              <span className="text-xs font-mono text-stone-500">
                {insights.predictions.ok?.toLocaleString() ?? 0} predictions
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 my-2">
              {Object.entries(conditionCounts).map(([label, count]) => (
                <div key={label} className="p-3.5 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8]">
                  <div className="flex items-center justify-between text-xs mb-1 text-[#173F32] font-semibold">
                    <span>{label}</span>
                    <span>{count} Plants</span>
                  </div>
                  <div className="w-full bg-[#DDE8D8] h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-[#2E8B57] h-full"
                      style={{ width: `${(count / Math.max(1, insights.explanationCoverage.totalPlants)) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Intelligence Navigation Cards to SHAP & Performance */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div
              onClick={() => onNavigate('shap-analysis')}
              className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
                <PieChart className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1">
                SHAP Explainability Workspace
              </h3>
              <p className="text-xs text-[#6B786E] leading-relaxed mb-4">
                Explore local waterfall attributions and global beeswarm feature impact distributions without black-box opacity.
              </p>
              <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                Open SHAP Workspace <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </div>

            <div
              onClick={() => onNavigate('model-performance')}
              className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
                <Target className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1">
                Model Performance & Matrix
              </h3>
              <p className="text-xs text-[#6B786E] leading-relaxed mb-4">
                Review holdout test confusion matrix, precision/recall, and per-class evaluation metrics.
              </p>
              <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                View Performance Metrics <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </div>

            <div
              onClick={() => onNavigate('recommendations')}
              className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card hover:border-[#2E8B57] transition-all cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-3">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-bold text-[#1C2B23] group-hover:text-[#2E8B57] transition-colors mb-1">
                Agronomic Recommendations
              </h3>
              <p className="text-xs text-[#6B786E] leading-relaxed mb-4">
                Rule-reviewed, actionable care directives based on empirical thresholds and species profiles.
              </p>
              <span className="text-xs font-semibold text-[#2E8B57] flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                Review Care Actions <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
