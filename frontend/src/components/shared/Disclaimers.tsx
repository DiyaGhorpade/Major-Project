import React from 'react';
import { Info } from 'lucide-react';
import { cn } from '../../lib/utils';
import { DISCLAIMERS, type DisclaimerKey } from './disclaimerText';

interface InfoTipProps {
  /** One of the required disclaimers, or custom text. */
  text: string;
  className?: string;
}

/** Small (i) icon with a native tooltip, for labels such as "Anomaly" or "Environmental state group 2". */
export const InfoTip: React.FC<InfoTipProps> = ({ text, className }) => (
  <span
    role="note"
    title={text}
    aria-label={text}
    tabIndex={0}
    className={cn('inline-flex items-center align-middle text-stone-400 hover:text-stone-600 cursor-help', className)}
  >
    <Info className="w-3.5 h-3.5" aria-hidden="true" />
  </span>
);

interface DisclaimerNoteProps {
  disclaimer: DisclaimerKey;
  className?: string;
}

/** Visible one-line note, e.g. under a recommendation card ("Advisory heuristic.") or on Model Performance. */
export const DisclaimerNote: React.FC<DisclaimerNoteProps> = ({ disclaimer, className }) => (
  <p className={cn('text-[11px] text-[#6B786E] italic leading-snug', className)}>{DISCLAIMERS[disclaimer]}</p>
);

export const AnomalyTip: React.FC<{ className?: string }> = ({ className }) => (
  <InfoTip text={DISCLAIMERS.anomaly} className={className} />
);

export const EnvStateTip: React.FC<{ className?: string }> = ({ className }) => (
  <InfoTip text={DISCLAIMERS.envState} className={className} />
);
