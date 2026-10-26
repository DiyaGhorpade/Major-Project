import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { AiInsightsPage } from './AiInsightsPage';
import { HealthContext, type HealthValue } from '../hooks/useHealth';
import { mockFetch } from '../test/helpers';
import { healthFixture, insightsFixture, metricsFixture } from '../test/fixtureHelpers';

afterEach(() => vi.unstubAllGlobals());

function renderPage(health: Partial<HealthValue> = {}, onNavigate = vi.fn()) {
  const value: HealthValue = { health: healthFixture(), error: null, mlConnected: true, mlUnreachable: false, ...health };
  return render(
    <HealthContext.Provider value={value}>
      <AiInsightsPage onNavigate={onNavigate} />
    </HealthContext.Provider>,
  );
}

describe('AiInsightsPage', () => {
  it('renders the real insights: model status, counts, labels', async () => {
    mockFetch({
      'GET /api/ai/insights': { body: insightsFixture() },
      'GET /api/models/performance': { body: metricsFixture() },
    });
    renderPage();

    const status = await screen.findByTestId('model-status-card');
    expect(within(status).getByText('AI model connected')).toBeInTheDocument();
    expect(within(status).getByText('v1.0.0')).toBeInTheDocument();
    expect(within(status).getByText('v1.0.0_20261021T145244Z')).toBeInTheDocument();
    expect(within(status).getByText('Available')).toBeInTheDocument();
    expect(within(status).getByText(/#1 completed: 5,376 of 5,376 readings predicted/)).toBeInTheDocument();

    // Anomalies: count and rate from the payload (102, 1.9%).
    expect(await screen.findByText('102')).toBeInTheDocument();
    expect(screen.getByText(/1\.9% \(anomaly records \/ ok predictions\)/)).toBeInTheDocument();
    expect(screen.queryAllByRole('note')).toHaveLength(0);

    // Explanation coverage and hold-out accuracy come from real data.
    expect(screen.getByText('16 / 16 Plants')).toBeInTheDocument();
    expect(screen.getByText('64.6%')).toBeInTheDocument();

    // Labels: stress-type guess, environmental state groups (never "health").
    expect(screen.getByText(/Stress-type guess/)).toBeInTheDocument();
    expect(screen.getByText('Environmental state group 1')).toBeInTheDocument();

    // Top global features come from the SHAP report.
    expect(screen.getAllByText('Soil moisture, 12-reading average').length).toBeGreaterThan(0);
    expect(screen.getByText('0.462')).toBeInTheDocument();
  });

  it('opens the anomalies page from the anomaly card', async () => {
    mockFetch({
      'GET /api/ai/insights': { body: insightsFixture() },
      'GET /api/models/performance': { body: metricsFixture() },
    });
    const onNavigate = vi.fn();
    renderPage({}, onNavigate);
    fireEvent.click(await screen.findByRole('button', { name: /Anomalous Readings/ }));
    expect(onNavigate).toHaveBeenCalledWith('anomalies');
  });

  it('shows none of the old hard-coded numbers', async () => {
    mockFetch({
      'GET /api/ai/insights': { body: insightsFixture() },
      'GET /api/models/performance': { body: metricsFixture() },
    });
    const { container } = renderPage();
    await screen.findByTestId('model-status-card');
    await screen.findByText('102');
    for (const fake of ['v1.2.0-jetson', '1.8 ms', '94.2%', '3 Attention Flags', '100% Ingest Coverage']) {
      expect(container.textContent).not.toContain(fake);
    }
  });

  it('shows the Run empty state before any analysis', async () => {
    const empty = insightsFixture();
    empty.hasPredictions = false;
    empty.lastPipelineRun = null;
    mockFetch({
      'GET /api/ai/insights': { body: empty },
      'GET /api/models/performance': { body: metricsFixture() },
    });
    renderPage();
    expect(await screen.findByTestId('run-empty-state')).toHaveTextContent('Press Run to analyze the plant data.');
    expect(screen.getByText('No analysis run yet')).toBeInTheDocument();
  });

  it('reports the AI model as unreachable', async () => {
    const empty = insightsFixture();
    empty.hasPredictions = false;
    const health = healthFixture();
    health.mlService.ok = false;
    mockFetch({
      'GET /api/ai/insights': { body: empty },
      'GET /api/models/performance': { body: metricsFixture() },
    });
    renderPage({ health, mlConnected: false, mlUnreachable: true });
    await waitFor(() => expect(screen.getAllByText('AI model unreachable').length).toBeGreaterThan(0));
    expect(screen.queryByTestId('run-empty-state')).not.toBeInTheDocument();
  });
});
