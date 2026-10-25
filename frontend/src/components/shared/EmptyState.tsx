import React, { ReactNode } from 'react';
import { Inbox, Play } from 'lucide-react';
import { cn } from '../../lib/utils';
import { usePipeline } from '../../hooks/usePipeline';
import { formatCount } from '../../lib/utils';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: React.ReactNode;
  actionText?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  actionText,
  onAction,
  className,
}) => {
  return (
    <div
      className={cn(
        'rounded-card border border-[#DCE4D9] bg-white p-10 text-center flex flex-col items-center justify-center max-w-lg mx-auto my-8 shadow-card',
        className
      )}
    >
      <div className="w-14 h-14 rounded-full bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-4">
        {icon || <Inbox className="w-7 h-7" />}
      </div>
      <h3 className="text-lg font-semibold text-[#1C2B23] mb-2">{title}</h3>
      <p className="text-sm text-[#6B786E] max-w-md leading-relaxed mb-6">
        {description}
      </p>
      {actionText && onAction && (
        <button
          onClick={onAction}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#2E8B57] hover:bg-[#236e44] text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
        >
          {actionText}
        </button>
      )}
    </div>
  );
};

interface RunEmptyStateProps {
  title?: string;
  className?: string;
}

/** Pre-run empty state: "Press Run to analyze the bundled plant dataset." with a Run action wired to the pipeline. */
export const RunEmptyState: React.FC<RunEmptyStateProps> = ({ title = 'No analysis yet', className }) => {
  const { run, isActive, hasCompletedRun, processed, total } = usePipeline();
  return (
    <div
      data-testid="run-empty-state"
      className={cn(
        'rounded-card border border-[#DCE4D9] bg-white p-10 text-center flex flex-col items-center justify-center max-w-lg mx-auto my-8 shadow-card',
        className
      )}
    >
      <div className="w-14 h-14 rounded-full bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center mb-4">
        <Play className="w-7 h-7" />
      </div>
      <h3 className="text-lg font-semibold text-[#1C2B23] mb-2">{title}</h3>
      <p className="text-sm text-[#6B786E] max-w-md leading-relaxed mb-6">
        Press <strong>Run</strong> to analyze the bundled plant dataset.
      </p>
      <button
        type="button"
        onClick={() => void run()}
        disabled={isActive}
        className="inline-flex items-center gap-2 px-4 py-2 bg-[#2E8B57] hover:bg-[#236e44] disabled:opacity-60 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
      >
        <Play className="w-4 h-4" aria-hidden="true" />
        <span>{isActive ? `Analyzing ${formatCount(processed)} / ${formatCount(total)}...` : hasCompletedRun ? 'Re-run' : 'Run'}</span>
      </button>
    </div>
  );
};
