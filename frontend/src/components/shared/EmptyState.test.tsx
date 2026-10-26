import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { RunEmptyState } from './EmptyState';
import { PipelineContext, idlePipeline, type PipelineValue } from '../../hooks/usePipeline';

function renderEmpty(value: Partial<PipelineValue> = {}) {
  return render(
    <PipelineContext.Provider value={{ ...idlePipeline, ...value }}>
      <RunEmptyState />
    </PipelineContext.Provider>,
  );
}

describe('RunEmptyState', () => {
  it('prompts to press Run and starts the pipeline', () => {
    const run = vi.fn().mockResolvedValue(undefined);
    renderEmpty({ run });
    expect(screen.getByTestId('run-empty-state')).toHaveTextContent('Press Run to analyze the plant data.');
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('disables the action and shows progress while a run is active', () => {
    renderEmpty({ isActive: true, state: 'running', processed: 10, total: 100 });
    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByRole('button')).toHaveTextContent('Analyzing 10 / 100');
  });
});
