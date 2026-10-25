import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { ApiError } from '../../api/client';
import { cn } from '../../lib/utils';

interface ErrorBannerProps {
  title?: string;
  /** Plain message, or an ApiError (its message is shown). */
  error: string | ApiError | null | undefined;
  onRetry?: () => void;
  retryLabel?: string;
  className?: string;
}

/** Shell-level banner for a failed run or an unreachable backend / ML service, with an optional retry. */
export const ErrorBanner: React.FC<ErrorBannerProps> = ({ title = 'Something went wrong', error, onRetry, retryLabel = 'Retry', className }) => {
  if (!error) return null;
  const message = typeof error === 'string' ? error : error.message;
  return (
    <div
      role="alert"
      data-testid="error-banner"
      className={cn(
        'flex flex-wrap items-center justify-between gap-3 px-4 md:px-6 py-2.5 bg-red-50 border-b border-red-200 text-xs text-red-900',
        className,
      )}
    >
      <div className="flex items-center gap-2 min-w-0">
        <AlertCircle className="w-4 h-4 shrink-0 text-red-600" aria-hidden="true" />
        <span className="font-semibold">{title}</span>
        <span className="text-red-800 truncate" title={message}>
          {message}
        </span>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-red-600 hover:bg-red-700 text-white font-medium transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" />
          <span>{retryLabel}</span>
        </button>
      )}
    </div>
  );
};
