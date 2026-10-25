import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AppShell } from './AppShell';
import { healthBody, mockFetch, statusBody } from '../../test/helpers';

afterEach(() => {
  vi.unstubAllGlobals();
});

const overviewBody = { recentAlerts: [], totals: {}, pipeline: statusBody() };

function renderShell() {
  return render(
    <AppShell currentRoute="overview" onNavigate={() => undefined}>
      <div>page content</div>
    </AppShell>,
  );
}

describe('AppShell', () => {
  it('always renders the synthetic-dataset disclaimer and the page', async () => {
    mockFetch({
      'GET /api/pipeline/status': { body: statusBody() },
      'GET /api/health': { body: healthBody() },
      'GET /api/monitoring/overview': { body: overviewBody },
    });
    renderShell();
    const banner = screen.getByTestId('synthetic-banner');
    expect(banner).toHaveTextContent('Synthetic dataset. Model outputs are experimental, not validated plant diagnoses.');
    expect(screen.getByText('page content')).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText('AI model connected').length).toBeGreaterThan(0));
    expect(screen.getByText('v1.0.0')).toBeInTheDocument();
    expect(screen.queryByTestId('error-banner')).not.toBeInTheDocument();
  });

  it('shows the failure cause in an error banner when the run failed', async () => {
    mockFetch({
      'GET /api/pipeline/status': {
        body: statusBody({ state: 'failed', mlServiceOk: false, message: 'ML service not reachable at :8001' }),
      },
      'GET /api/health': { body: healthBody() },
      'GET /api/monitoring/overview': { body: overviewBody },
    });
    renderShell();
    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('Analysis run failed');
    expect(banner).toHaveTextContent('ML service not reachable at :8001');
    expect(screen.getByRole('button', { name: 'Run again' })).toBeInTheDocument();
    expect(screen.getByTestId('synthetic-banner')).toBeInTheDocument();
  });

  it('warns when health says the ML service is unreachable', async () => {
    mockFetch({
      'GET /api/pipeline/status': { body: statusBody({ mlServiceOk: false }) },
      'GET /api/health': {
        body: healthBody({ mlService: { ok: false, url: 'http://127.0.0.1:8001', detail: 'connection refused', status: null, modelsLoaded: null, explainabilityAvailable: null } }),
      },
      'GET /api/monitoring/overview': { body: overviewBody },
    });
    renderShell();
    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('ML service not reachable at http://127.0.0.1:8001');
    expect((await screen.findAllByText('AI model unreachable')).length).toBeGreaterThan(0);
  });

  it('reports an unreachable backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    renderShell();
    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('Backend not reachable');
  });
});
