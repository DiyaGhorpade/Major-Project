import React from 'react';
import { Database } from 'lucide-react';
import type { DatasetLabels } from '../../types';
import { orNA } from '../../lib/utils';

/** Dataset labels, always shown apart from model output. */
export const DatasetLabelsBlock: React.FC<{ labels: DatasetLabels | null | undefined; className?: string }> = ({ labels, className }) => (
  <div data-testid="dataset-labels" className={`rounded-xl border border-stone-300 bg-stone-50 p-3.5 ${className ?? ''}`}>
    <h4 className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-stone-700">
      <Database className="h-3.5 w-3.5" aria-hidden="true" />
      Dataset labels
    </h4>
    <dl className="grid grid-cols-3 gap-2 text-xs">
      <div>
        <dt className="text-[10px] uppercase text-stone-400">Stress level</dt>
        <dd className="font-semibold text-[#1C2B23]">{orNA(labels?.stressLevel)}</dd>
      </div>
      <div>
        <dt className="text-[10px] uppercase text-stone-400">Stress type</dt>
        <dd className="font-semibold text-[#1C2B23]">{orNA(labels?.stressType)}</dd>
      </div>
      <div>
        <dt className="text-[10px] uppercase text-stone-400">Anomaly label</dt>
        <dd className="font-semibold text-[#1C2B23]">
          {labels?.isSyntheticAnomaly === null || labels?.isSyntheticAnomaly === undefined ? 'n/a' : labels.isSyntheticAnomaly ? 'yes' : 'no'}
        </dd>
      </div>
    </dl>
  </div>
);
