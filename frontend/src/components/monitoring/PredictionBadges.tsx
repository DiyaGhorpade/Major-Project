import React from 'react';
import type { PlantHealthStatus, PlantPrediction, Severity } from '../../types';
import { StatusBadge } from '../shared/StatusBadge';
import { AnomalyTip, EnvStateTip, InfoTip } from '../shared/Disclaimers';
import { DISCLAIMERS } from '../shared/disclaimerText';
import { cn } from '../../lib/utils';
import { formatConfidence } from './format';
import { SEVERITY_LABELS } from '../../lib/statusLabels';

/** Plant status from the model's severity (ML), not from the experimental group. Copy lives in lib/statusLabels. */
export const PlantStatusBadge: React.FC<{ status: PlantHealthStatus | string; size?: 'sm' | 'md' }> = ({ status, size = 'sm' }) => (
  <StatusBadge status={status} size={size} />
);

const SEVERITY_CLASSES: Record<Severity, string> = {
  healthy: 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]',
  low: 'bg-yellow-50 text-yellow-800 border-yellow-200',
  moderate: 'bg-amber-50 text-amber-800 border-amber-300',
  high: 'bg-red-50 text-red-700 border-red-200',
};

export const SeverityBadge: React.FC<{ severity: Severity | null | undefined; className?: string }> = ({ severity, className }) => {
  if (!severity) {
    return (
      <span className={cn('inline-flex rounded-full border border-stone-200 bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-600', className)}>
        n/a
      </span>
    );
  }
  const cls = SEVERITY_CLASSES[severity] ?? 'bg-stone-100 text-stone-700 border-stone-200';
  return <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-xs font-medium', cls, className)}>{SEVERITY_LABELS[severity] ?? severity}</span>;
};

export const AnomalyFlag: React.FC<{ isAnomaly: boolean | null | undefined }> = ({ isAnomaly }) => (
  <span className="inline-flex items-center gap-1">
    {isAnomaly === null || isAnomaly === undefined ? (
      <span className="text-stone-500">Anomaly check: n/a</span>
    ) : isAnomaly ? (
      <span className="font-semibold text-amber-800">Unusual reading</span>
    ) : (
      <span className="text-stone-600">Normal reading</span>
    )}
    <AnomalyTip />
  </span>
);

export const EnvStateLabel: React.FC<{ envState: PlantPrediction['envState'] | undefined }> = ({ envState }) => (
  <span className="inline-flex items-center gap-1">
    <span className={envState ? 'text-stone-700' : 'text-stone-500'}>{envState ? envState.label : 'Environmental state: n/a'}</span>
    <EnvStateTip />
  </span>
);

interface PredictionSummaryProps {
  prediction: PlantPrediction | null | undefined;
  className?: string;
}

/** Compact model-output block for plant cards: severity (ML), stress type (heuristic), env state, anomaly. */
export const PredictionSummary: React.FC<PredictionSummaryProps> = ({ prediction, className }) => {
  if (!prediction) {
    return (
      <div data-testid="prediction-missing" className={cn('rounded-lg border border-dashed border-stone-300 bg-stone-50 px-2.5 py-2 text-[11px] text-stone-500', className)}>
        Not analyzed yet. {DISCLAIMERS.runPrompt}
      </div>
    );
  }
  return (
    <div className={cn('space-y-1.5 rounded-lg border border-[#DDE8D8] bg-[#EDF4EB]/50 px-2.5 py-2 text-[11px]', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-stone-500">Severity (ML)</span>
        <span className="flex items-center gap-1.5">
          <SeverityBadge severity={prediction.severity} />
          <span className="font-mono text-stone-600">{formatConfidence(prediction.confidence)}</span>
        </span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1 text-stone-500">
          Stress type guess <InfoTip text={DISCLAIMERS.conditionGuess} />
        </span>
        <span className="font-semibold text-[#1C2B23]">{prediction.condition ?? 'n/a'}</span>
      </div>
      <div className="text-stone-600">
        <EnvStateLabel envState={prediction.envState} />
      </div>
      <div>
        <AnomalyFlag isAnomaly={prediction.anomaly?.isAnomaly} />
      </div>
    </div>
  );
};
