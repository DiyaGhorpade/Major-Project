import { afterEach, describe, expect, it, vi } from 'vitest';
import { StrictMode } from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { DataLogsPage } from './DataLogsPage';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, flaggedFixture, monitoringFixture, plantsFixture, predictionFixture, readingsFixture, sessionsFixture } from '../test/monitoringFixtures';

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

function routes(logs: unknown = readingsFixture(), total = '5376') {
  return {
    'GET /api/data-logs': { body: logs, headers: { 'X-Total-Count': total } },
    'GET /api/plants': { body: plantsFixture(), headers: { 'X-Total-Count': '4' } },
    'GET /api/sessions': { body: sessionsFixture(), headers: { 'X-Total-Count': '4' } },
  };
}

describe('DataLogsPage', () => {
  it('renders server-paginated rows, the total, status and validation_error', async () => {
    mockFetch(routes([...flaggedFixture(), ...readingsFixture().slice(0, 2)], '233'));
    render(<DataLogsPage />);

    const flagged = (await screen.findAllByText('missing: soil'))[0].closest('tr')!;
    expect(within(flagged).getByText('Flagged')).toBeInTheDocument();
    expect(within(flagged).getAllByText('n/a').length).toBeGreaterThan(0); // soil is null, shown as n/a not 0
    expect(screen.getByText(/233 Records Match/)).toBeInTheDocument();
    expect(screen.getByText(/Page 1 of 16/)).toBeInTheDocument();
    expect(screen.queryByText(/Jetson/)).not.toBeInTheDocument();
  });

  it('offers a flagged status filter that is sent to the API and to the CSV download', async () => {
    const { calls } = mockFetch(routes());
    render(<DataLogsPage />);
    await screen.findAllByText('Pot #16');

    const status = screen.getByLabelText('Status filter');
    expect(within(status).getByRole('option', { name: /Flagged/ })).toBeInTheDocument();
    fireEvent.change(status, { target: { value: 'flagged' } });

    await waitFor(() => expect(calls.some((c) => c.url.includes('/api/data-logs') && c.url.includes('status=flagged'))).toBe(true));
    const link = screen.getByRole('link', { name: /Download Filtered CSV/ });
    expect(link.getAttribute('href')).toContain('format=csv');
    expect(link.getAttribute('href')).toContain('status=flagged');
    expect(link.getAttribute('href')).not.toContain('limit=');
  });

  it('applies a session handed over from the Sessions page', async () => {
    window.sessionStorage.setItem('plant-app.dataLogs.sessionId', '2');
    const { calls } = mockFetch(routes());
    render(<DataLogsPage />);
    await screen.findAllByText('Pot #16');
    expect(calls.some((c) => c.url.includes('sessionId=2'))).toBe(true);
    expect(screen.getByRole('link', { name: /Download Filtered CSV/ }).getAttribute('href')).toContain('sessionId=2');
  });

  it('the record drawer shows ML output and dataset labels as separate sections', async () => {
    const record = monitoringFixture().plant.latestReading!;
    mockFetch({
      ...routes([record], '1'),
      'GET /api/predictions': { body: predictionFixture(), headers: { 'X-Total-Count': '1' } },
    });
    render(<DataLogsPage />);
    fireEvent.click((await screen.findAllByText('Pot #5'))[0]);

    const ml = await screen.findByTestId('record-ml');
    await within(ml).findByText('High stress');
    expect(ml).toHaveTextContent('Water Stress');
    expect(ml).toHaveTextContent('Environmental state group 1');
    const labels = screen.getByTestId('dataset-labels');
    expect(labels).toHaveTextContent('Dataset labels');
    expect(ml).not.toContainElement(labels);
    expect(labels).not.toHaveTextContent('Environmental state');
  });

  it('the drawer says there is no model output before a Run, and shows validation_error for flagged rows', async () => {
    mockFetch({ ...routes(flaggedFixture(), '233'), 'GET /api/predictions': { body: [], headers: { 'X-Total-Count': '0' } } });
    render(<DataLogsPage />);
    fireEvent.click((await screen.findAllByText('Pot #16'))[0]);
    expect(await screen.findByText(/No model output for this reading yet\. Press Run/)).toBeInTheDocument();
    expect(screen.getByTestId('validation-error')).toHaveTextContent('missing: soil');
  });

  it('keeps the handed-over session under React.StrictMode (initializers run twice) and consumes it once', async () => {
    window.sessionStorage.setItem('plant-app.dataLogs.sessionId', '3');
    const { calls } = mockFetch(routes());
    const { unmount } = render(
      <StrictMode>
        <DataLogsPage />
      </StrictMode>,
    );
    await screen.findAllByText('Pot #16');
    await waitFor(() => expect(calls.some((c) => c.url.includes('/api/data-logs') && c.url.includes('sessionId=3'))).toBe(true));
    expect(screen.getByLabelText('Session filter')).toHaveValue('3');
    expect(window.sessionStorage.getItem('plant-app.dataLogs.sessionId')).toBeNull();
    unmount();

    // Coming back later is unfiltered.
    render(
      <StrictMode>
        <DataLogsPage />
      </StrictMode>,
    );
    await screen.findAllByText('Pot #16');
    expect(screen.getByLabelText('Session filter')).toHaveValue('all');
  });

  it('shows timestamps as formatted IST (raw ISO only in the tooltip)', async () => {
    mockFetch(routes());
    render(<DataLogsPage />);
    const cell = (await screen.findAllByText('Oct 18, 22:05:15'))[0];
    expect(cell).toHaveAttribute('title', '2026-10-18T22:05:15+05:30');
    expect(screen.queryAllByText('2026-10-18T22:05:15+05:30')).toHaveLength(0);
  });

  it('shows the API error', async () => {
    mockFetch({ ...routes(), 'GET /api/data-logs': { status: 500, body: errorEnvelope('internal_error', 'Logs unavailable') } });
    render(<DataLogsPage />);
    expect(await screen.findByText('Logs unavailable')).toBeInTheDocument();
  });
});
