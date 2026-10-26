import React from 'react';
import {
  Target,
  CheckCircle2,
} from 'lucide-react';
import { useModelMetrics } from '../hooks/useData';
import { ErrorState } from '../components/shared/ErrorState';
import { MetricCard } from '../components/shared/MetricCard';
import { ConfusionMatrixChart } from '../components/charts/ConfusionMatrixChart';
import { EmptyState } from '../components/shared/EmptyState';
import { formatDateTime, formatFraction } from '../lib/utils';
import type { ModelMetrics } from '../types';

const num = (v: unknown, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : 'n/a');
const asRecord = (v: unknown): Record<string, unknown> | null => (typeof v === 'object' && v !== null ? (v as Record<string, unknown>) : null);
const str = (v: unknown): string | null => (typeof v === 'string' && v.length > 0 ? v : null);

const BaselineTable: React.FC<{ metrics: ModelMetrics }> = ({ metrics }) => {
  const names: Record<string, string> = { majority_class: 'Majority-class guess', logistic_regression: 'Logistic regression' };
  const rows = [
    { key: 'model', label: `${metrics.model_name} (this model)`, accuracy: metrics.overall.accuracy, macroF1: metrics.overall.f1_score, isModel: true },
    ...Object.entries(metrics.baselines ?? {}).map(([key, b]) => ({
      key,
      label: names[key] ?? key.replace(/_/g, ' '),
      accuracy: b.accuracy,
      macroF1: b.macro_f1,
      isModel: false,
    })),
  ];
  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden" data-testid="baselines">
      <div className="p-4 border-b border-[#DCE4D9]">
        <h3 className="text-sm font-bold text-[#1C2B23]">Comparison with simple baselines</h3>
        <p className="text-xs text-[#6B786E]">Same hold-out rows, same labels.</p>
      </div>
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
            <th className="py-3 px-4 font-semibold">Method</th>
            <th className="py-3 px-4 font-semibold">Accuracy</th>
            <th className="py-3 px-4 font-semibold">Macro F1</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#DCE4D9]/60">
          {rows.map((r) => (
            <tr key={r.key} className={r.isModel ? 'bg-[#EDF4EB]/50' : ''}>
              <td className="py-3 px-4 font-bold text-[#1C2B23]">{r.label}</td>
              <td className="py-3 px-4 font-mono">{formatFraction(r.accuracy)}</td>
              <td className="py-3 px-4 font-mono">{formatFraction(r.macroF1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const AnomalySummary: React.FC<{ model: Record<string, unknown> }> = ({ model }) => {
  const vs = asRecord(model.metricsVsHelperLabel);
  const threshold = asRecord(model.threshold);
  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card" data-testid="anomaly-model">
      <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-1.5 mb-1">
        Anomaly model (Isolation Forest)
      </h3>
      <p className="text-xs text-[#6B786E] mb-3">Flags unusual sensor readings.</p>
      {vs && (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
          <div>
            <dt className="text-[#6B786E]">Precision vs helper label</dt>
            <dd className="font-mono font-semibold">{formatFraction(vs.precision as number)}</dd>
          </div>
          <div>
            <dt className="text-[#6B786E]">Recall vs helper label</dt>
            <dd className="font-mono font-semibold">{formatFraction(vs.recall as number)}</dd>
          </div>
          <div>
            <dt className="text-[#6B786E]">F1</dt>
            <dd className="font-mono font-semibold">{formatFraction(vs.f1 as number)}</dd>
          </div>
          <div>
            <dt className="text-[#6B786E]">ROC AUC / PR AUC</dt>
            <dd className="font-mono font-semibold">
              {num(vs.roc_auc)} / {num(vs.pr_auc)}
            </dd>
          </div>
          <div>
            <dt className="text-[#6B786E]">Flagged / helper positives</dt>
            <dd className="font-mono font-semibold">
              {String(vs.n_flagged ?? 'n/a')} / {String(vs.n_helper_positives ?? 'n/a')} of {String(vs.n_samples ?? 'n/a')} rows
            </dd>
          </div>
          {threshold && (
            <div>
              <dt className="text-[#6B786E]">Decision rule</dt>
              <dd className="font-mono font-semibold">{str(threshold.decision_rule) ?? 'n/a'}</dd>
            </div>
          )}
        </dl>
      )}
    </div>
  );
};

const ClusteringSummary: React.FC<{ model: Record<string, unknown> }> = ({ model }) => (
  <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card" data-testid="clustering-model">
    <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-1.5 mb-1">
      Environmental states (KMeans)
    </h3>
    <p className="text-xs text-[#6B786E] mb-3">Groups of similar environmental readings.</p>
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
      <div>
        <dt className="text-[#6B786E]">Number of environmental state groups</dt>
        <dd className="font-mono font-semibold">{String(model.selectedK ?? 'n/a')}</dd>
      </div>
      <div>
        <dt className="text-[#6B786E]">Silhouette score</dt>
        <dd className="font-mono font-semibold">{num(model.silhouette)}</dd>
      </div>
    </dl>
  </div>
);

export const ModelPerformancePage: React.FC = () => {
  const metricsQuery = useModelMetrics();
  const metrics = metricsQuery.data;
  const classCount = metrics?.confusion_matrix.labels.length ?? 0;

  if (metricsQuery.error && !metrics) {
    return <ErrorState message={metricsQuery.error.message} onRetry={() => void metricsQuery.refetch()} />;
  }

  if (!metrics) {
    return (
      <EmptyState
        icon={<Target className="w-8 h-8 text-stone-400" />}
        title={metricsQuery.loading ? 'Loading evaluation metrics...' : 'Evaluation Metrics Unavailable'}
        description={metricsQuery.loading ? '' : 'No evaluation artifacts found for the active model version.'}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header & Artifact Specification */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              Offline Benchmark Evaluation
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              {metrics.model_version}
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            {metrics.model_name}
          </h1>
          <p className="text-xs text-[#6B786E] mt-1 max-w-2xl">
            Evaluated on a hold-out set: {metrics.split}. {metrics.total_samples.toLocaleString()} samples.
          </p>
          <div className="flex flex-wrap items-center gap-2 mt-2">
            <span className="text-[11px] text-stone-500 font-mono">Run {metrics.runId}</span>
          </div>
        </div>

        <div className="text-right text-xs text-stone-500 font-mono">
          <div>Evaluated: {formatDateTime(metrics.evaluated_at)} IST</div>
          <div className="text-stone-400 text-[11px] mt-0.5">Dataset: {metrics.dataset_name}</div>
        </div>
      </div>

      {/* Global Evaluation Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <MetricCard
          title="Accuracy"
          value={formatFraction(metrics.overall.accuracy)}
          subtitle="Top-1, hold-out set"
          icon={<Target className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Macro Precision"
          value={formatFraction(metrics.overall.precision)}
          subtitle={`Mean over ${classCount} classes`}
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Macro Recall"
          value={formatFraction(metrics.overall.recall)}
          subtitle={`Mean over ${classCount} classes`}
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Macro F1"
          value={formatFraction(metrics.overall.f1_score)}
          subtitle={`Mean over ${classCount} classes`}
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Balanced Accuracy"
          value={formatFraction(metrics.balancedAccuracy)}
          subtitle="Mean per-class recall"
          icon={<Target className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Weighted F1"
          value={formatFraction(metrics.weightedF1)}
          subtitle={`Weighted by class support (${classCount} classes)`}
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
      </div>

      {/* Confusion Matrix Chart */}
      <ConfusionMatrixChart metrics={metrics} />

      <BaselineTable metrics={metrics} />

      {/* Per-Class Metrics Detailed Table */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="p-4 border-b border-[#DCE4D9] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              Per-class evaluation
            </h3>
            <p className="text-xs text-[#6B786E]">
              Precision, recall and F1 for each model-rated severity class.
            </p>
          </div>
          <span className="text-xs font-mono text-stone-500">
            {classCount} classes
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">Severity class</th>
                <th className="py-3 px-4 font-semibold">Precision</th>
                <th className="py-3 px-4 font-semibold">Recall (Sensitivity)</th>
                <th className="py-3 px-4 font-semibold">F1-Score</th>
                <th className="py-3 px-4 font-semibold">Support (test rows)</th>
                <th className="py-3 px-4 font-semibold text-right">How reliable</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60">
 {metrics.per_class.map((c) => (
                <tr key={c.class_name} className="hover:bg-stone-50/50">
                  <td className="py-3 px-4 font-bold text-[#1C2B23]">
                    {c.class_name}
                  </td>
                  <td className="py-3 px-4 font-mono font-medium text-stone-800">
                    {formatFraction(c.precision)}
                  </td>
                  <td className="py-3 px-4 font-mono font-medium text-stone-800">
                    {formatFraction(c.recall)}
                  </td>
                  <td className="py-3 px-4 font-mono font-bold text-[#2E8B57]">
                    {formatFraction(c.f1_score)}
                  </td>
                  <td className="py-3 px-4 font-mono text-stone-600">
                    {c.support.toLocaleString()}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold bg-[#EDF4EB] text-[#2E8B57] border border-[#DDE8D8]">
                      {c.laymanStatus}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {(metrics.anomalyModel || metrics.clusteringModel) && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {metrics.anomalyModel && <AnomalySummary model={metrics.anomalyModel} />}
          {metrics.clusteringModel && <ClusteringSummary model={metrics.clusteringModel} />}
        </div>
      )}
    </div>
  );
};
