import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { SessionsPage } from './SessionsPage';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, plantsFixture, sessionsFixture, settingsFixture } from '../test/monitoringFixtures';

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

// The data has 16 plants; the fixture holds 4, so repeat them to the real count.
const sixteenPlants = () => Array.from({ length: 16 }, (_, i) => ({ ...plantsFixture()[i % 4], id: i + 1 }));

const routes = () => ({
  'GET /api/plants': { body: sixteenPlants(), headers: { 'X-Total-Count': '16' } },
  'GET /api/sessions': { body: sessionsFixture(), headers: { 'X-Total-Count': '4' } },
  'GET /api/settings': { body: settingsFixture() },
});

describe('SessionsPage', () => {
  it('lists real sessions, derives the schedule from settings and shows flagged counts', async () => {
    mockFetch(routes());
    render(<SessionsPage onNavigate={vi.fn()} />);

    expect(await screen.findByTestId('session-row-4')).toBeInTheDocument();
    expect(screen.getByText('4 sessions logged')).toBeInTheDocument();
    // Observed from the data (1,344 readings / 16 plants over ~7 days) and the configured schedule, labelled apart.
    expect(await screen.findByTestId('observed-cadence')).toHaveTextContent('Data shows readings every 2 h (12 per day)');
    expect(screen.getByTestId('configured-schedule')).toHaveTextContent('Configured schedule: every 2 h (12 per day)');
    expect(screen.queryByText(/3x Daily/)).not.toBeInTheDocument();
    expect(screen.queryByText(/08:00, 13:00, 18:00/)).not.toBeInTheDocument();
    expect(within(screen.getByTestId('session-row-4')).getByText('56 flagged')).toBeInTheDocument();
    expect(within(screen.getByTestId('session-row-4')).getByText('1,344 / 1,344')).toBeInTheDocument();
    expect(within(screen.getByTestId('session-row-4')).getByText(/Span: 6d 22h/)).toBeInTheDocument();

    // Selected session (newest) shows per-node progress.
    const detail = screen.getByTestId('session-detail');
    expect(within(detail).getByTestId('node-progress-1')).toHaveTextContent('672 / 672 readings');
    expect(within(detail).getByTestId('node-progress-2')).toHaveTextContent('Node 2');
    expect(within(detail).getByTestId('flagged-count')).toHaveTextContent('56');

    fireEvent.click(screen.getByTestId('session-row-2'));
    expect(screen.getByTestId('flagged-count')).toHaveTextContent('71');
    expect(within(screen.getByTestId('session-detail')).getByText('Session #2')).toBeInTheDocument();
  });

  it('hands the chosen session to the Data Logs page', async () => {
    const onNavigate = vi.fn();
    mockFetch(routes());
    render(<SessionsPage onNavigate={onNavigate} />);
    fireEvent.click(await screen.findByTestId('session-row-3'));
    fireEvent.click(screen.getByRole('button', { name: 'Filter Records by This Session' }));
    expect(onNavigate).toHaveBeenCalledWith('data-logs');
    expect(window.sessionStorage.getItem('plant-app.dataLogs.sessionId')).toBe('3');
  });

  it('shows empty and error states', async () => {
    mockFetch({ ...routes(), 'GET /api/sessions': { body: [], headers: { 'X-Total-Count': '0' } } });
    const { unmount } = render(<SessionsPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('No sessions recorded.')).toBeInTheDocument();
    unmount();

    mockFetch({ ...routes(), 'GET /api/sessions': { status: 500, body: errorEnvelope('internal_error', 'Sessions down') } });
    render(<SessionsPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('Sessions down')).toBeInTheDocument();
  });
});
