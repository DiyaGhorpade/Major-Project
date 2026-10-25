import React from 'react';
import {
  Target,
  CheckCircle2,
} from 'lucide-react';
import { useModelMetrics } from '../hooks/useData';
import { ErrorState } from '../components/shared/ErrorState';
import { DisclaimerNote } from '../components/shared/Disclaimers';
import { MetricCard } from '../components/shared/MetricCard';
import { ConfusionMatrixChart } from '../components/charts/ConfusionMatrixChart';
import { EmptyState } from '../components/shared/EmptyState';
import { formatDateTime } from '../lib/utils';

export const ModelPerformancePage: React.FC = () => {
  const metricsQuery = useModelMetrics();
  const metrics = metricsQuery.data;

  if (metricsQuery.error) {
    return <ErrorState message={metricsQuery.error.message} onRetry={() => void metricsQuery.refetch()} />;
  }

  if (!metrics) {
    return (
      <EmptyState
        icon={<Target className="w-8 h-8 text-stone-400" />}
        title={metricsQuery.loading ? "Loading evaluation metrics..." : "Evaluation Benchmark Metrics Unavailable"}
        description={metricsQuery.loading ? "" : "No evaluation artifacts found for the active model version."}
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
            Evaluated on a chronological hold-out set ({metrics.split}) consisting of {metrics.total_samples.toLocaleString()} samples.
          </p>
        </div>

        <div className="text-right text-xs text-stone-500 font-mono">
          <div>Evaluated: {formatDateTime(metrics.evaluated_at)}</div>
          <div className="text-stone-400 text-[11px] mt-0.5">Dataset: {metrics.dataset_name}</div>
        </div>
      </div>

      {/* Global Evaluation Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Overall Test Accuracy"
          value={`${(metrics.overall.accuracy * 100).toFixed(1)}%`}
          subtitle="Top-1 multi-class classification"
          icon={<Target className="w-4 h-4 text-[#2E8B57]" />}
          trend={{ value: 'Holdout test', direction: 'up' }}
        />
        <MetricCard
          title="Macro Precision"
          value={`${(metrics.overall.precision * 100).toFixed(1)}%`}
          subtitle={`Macro average over ${metrics.confusion_matrix.labels.length} classes`}
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="Macro Recall"
          value={`${(metrics.overall.recall * 100).toFixed(1)}%`}
          subtitle={`Macro average over ${metrics.confusion_matrix.labels.length} classes`}
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
        <MetricCard
          title="F1-Score (Harmonic Mean)"
          value={`${(metrics.overall.f1_score * 100).toFixed(1)}%`}
          subtitle="Balanced precision-recall score"
          icon={<CheckCircle2 className="w-4 h-4 text-[#2E8B57]" />}
        />
      </div>

      <DisclaimerNote disclaimer="modelPerformance" />

      {/* Confusion Matrix Chart */}
      <ConfusionMatrixChart metrics={metrics} />

      {/* Per-Class Metrics Detailed Table */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="p-4 border-b border-[#DCE4D9] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              Per-Phenotype Class Evaluation Breakdown
            </h3>
            <p className="text-xs text-[#6B786E]">
              Precision, recall, and F1 scores by individual plant condition category.
            </p>
          </div>
          <span className="text-xs font-mono text-stone-500">
            {metrics.confusion_matrix.labels.length} Classes
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">Phenotype / Condition Class</th>
                <th className="py-3 px-4 font-semibold">Precision</th>
                <th className="py-3 px-4 font-semibold">Recall (Sensitivity)</th>
                <th className="py-3 px-4 font-semibold">F1-Score</th>
                <th className="py-3 px-4 font-semibold">Support (Test Samples)</th>
                <th className="py-3 px-4 font-semibold text-right">Reliability Index</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60">
              {metrics.per_class.map((c, idx) => (
                <tr key={idx} className="hover:bg-stone-50/50">
                  <td className="py-3 px-4 font-bold text-[#1C2B23]">
                    {c.class_name}
                  </td>
                  <td className="py-3 px-4 font-mono font-medium text-stone-800">
                    {(c.precision * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 font-mono font-medium text-stone-800">
                    {(c.recall * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 font-mono font-bold text-[#2E8B57]">
                    {(c.f1_score * 100).toFixed(1)}%
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
    </div>
  );
};
