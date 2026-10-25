import React, { useState } from 'react';
import { FlaskConical, X } from 'lucide-react';
import { DISCLAIMERS } from './Disclaimers';

/**
 * Persistent provenance notice shown on every page. The full banner can be collapsed, but a compact
 * chip with the same text always remains visible (the disclaimer is never fully dismissable).
 */
export const SyntheticBanner: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <div role="note" data-testid="synthetic-banner" className="px-4 md:px-6 py-1 bg-white border-b border-[#DCE4D9]">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          title={DISCLAIMERS.synthetic}
          className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-[11px] font-medium text-amber-900"
        >
          <FlaskConical className="w-3 h-3" aria-hidden="true" />
          <span>{DISCLAIMERS.synthetic}</span>
        </button>
      </div>
    );
  }

  return (
    <div
      role="note"
      data-testid="synthetic-banner"
      className="flex items-center justify-between gap-3 px-4 md:px-6 py-2 bg-amber-50/90 border-b border-amber-200/80 text-xs text-amber-900"
    >
      <div className="flex items-center gap-2">
        <FlaskConical className="w-4 h-4 shrink-0 text-amber-700" aria-hidden="true" />
        <span className="font-medium">{DISCLAIMERS.synthetic}</span>
      </div>
      <button
        type="button"
        onClick={() => setCollapsed(true)}
        aria-label="Collapse notice to a chip"
        className="p-1 rounded text-amber-700 hover:bg-amber-100"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};
