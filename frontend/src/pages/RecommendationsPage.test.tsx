import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { RecommendationsPage } from './RecommendationsPage';
import { jsonResponse, type MockRoute } from '../test/helpers';
import { errorEnvelope, plantsFixture, recommendationsFixture } from '../test/fixtureHelpers';
import type { Recommendation } from '../types';

afterEach(() => vi.unstubAllGlobals());

interface Handlers {
  list?: (params: URLSearchParams) => MockRoute;
  patch?: (id: string, body: { status: string }) => MockRoute | Promise<MockRoute>;
}

/** Fetch stub that sees query strings (the shared helper ignores them). */
function stubBackend(handlers: Handlers) {
  const urls: string[] = [];
  const patches: Array<{ id: string; status: string }> = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost');
      urls.push(`${init?.method ?? 'GET'} ${url.pathname}${url.search}`);
      if (url.pathname === '/api/plants') return jsonResponse({ body: plantsFixture(), headers: { 'X-Total-Count': '16' } });
      if (url.pathname === '/api/recommendations' && (init?.method ?? 'GET') === 'GET') {
        return jsonResponse(handlers.list ? handlers.list(url.searchParams) : { body: recommendationsFixture(), headers: { 'X-Total-Count': '2' } });
      }
      const m = url.pathname.match(/^\/api\/recommendations\/(.+)$/);
      if (m && init?.method === 'PATCH') {
        const body = JSON.parse(String(init.body)) as { status: string };
        patches.push({ id: m[1], status: body.status });
        return jsonResponse(await (handlers.patch ? handlers.patch(m[1], body) : { body: { ...recommendationsFixture()[0], status: body.status } }));
      }
      return jsonResponse({ status: 404, body: errorEnvelope('not_found', url.pathname) });
    }),
  );
  return { urls, patches };
}

const renderPage = () => render(<RecommendationsPage onNavigate={vi.fn()} />);

describe('RecommendationsPage', () => {
  it('lists server-paginated recommendations with real status counts and no advisory disclaimer', async () => {
    const { urls } = stubBackend({
      list: (p) => {
        const status = p.get('status');
        if (status) {
          const totals: Record<string, string> = { new: '7', in_progress: '1', completed: '3', dismissed: '2', superseded: '4' };
          return { body: [], headers: { 'X-Total-Count': totals[status] } };
        }
        return { body: recommendationsFixture(), headers: { 'X-Total-Count': '7' } };
      },
    });
    renderPage();
    expect(await screen.findAllByText('Check watering')).toHaveLength(1);
    expect(screen.getByText('Monitor and inspect')).toBeInTheDocument();
    expect(screen.queryAllByTestId('rec-disclaimer')).toHaveLength(0);
    expect(screen.getByTestId('range-label')).toHaveTextContent('Showing 1-7 of 7 directives');

    await waitFor(() => expect(screen.getByTestId('count-new')).toHaveTextContent('7'));
    expect(screen.getByTestId('count-in_progress')).toHaveTextContent('1');
    expect(screen.getByTestId('count-completed')).toHaveTextContent('3');
    expect(screen.getByTestId('count-dismissed')).toHaveTextContent('2');
    expect(screen.getByTestId('count-superseded')).toHaveTextContent('4');

    // Server-side pagination, not "first 500 then filter on the client".
    const listCall = urls.find((u) => u.startsWith('GET /api/recommendations') && !u.includes('status='));
    expect(listCall).toContain('limit=12');
    expect(listCall).not.toContain('limit=500');
  });

  it('lists the pots with no open recommendation so every pot is visible', async () => {
    stubBackend({
      list: (p) => {
        const status = p.get('status');
        if (status === 'new') return { body: recommendationsFixture(), headers: { 'X-Total-Count': '2' } }; // pots 16 and 8
        if (status) return { body: [], headers: { 'X-Total-Count': '0' } };
        return { body: recommendationsFixture(), headers: { 'X-Total-Count': '2' } };
      },
    });
    renderPage();
    const strip = await screen.findByTestId('no-action-plants');
    expect(strip).toHaveTextContent('14 of 16 pots');
    expect(strip).toHaveTextContent(/Pot 1: /);
    expect(strip).not.toHaveTextContent(/Pot 8: /);
    expect(strip).not.toHaveTextContent(/Pot 16: /);

    fireEvent.change(screen.getByLabelText('Priority filter'), { target: { value: 'high' } });
    await waitFor(() => expect(screen.queryByTestId('no-action-plants')).not.toBeInTheDocument());
  });

  it('sends every filter to the server and returns to the first page', async () => {
    const { urls } = stubBackend({ list: () => ({ body: recommendationsFixture(), headers: { 'X-Total-Count': '30' } }) });
    renderPage();
    await screen.findByText('Check watering');
    await screen.findByRole('option', { name: /^Pot 16: / });

    fireEvent.click(screen.getByRole('button', { name: /Next/ }));
    await waitFor(() => expect(urls.some((u) => u.includes('offset=12'))).toBe(true));
    expect(await screen.findByText('Page 2 of 3')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Status filter'), { target: { value: 'new' } });
    fireEvent.change(screen.getByLabelText('Priority filter'), { target: { value: 'high' } });
    fireEvent.change(screen.getByLabelText('Category filter'), { target: { value: 'sensor_check' } });
    fireEvent.change(screen.getByLabelText('Source filter'), { target: { value: 'heuristic' } });
    fireEvent.change(screen.getByLabelText('Plant filter'), { target: { value: '8' } });
    fireEvent.click(screen.getByLabelText('Show superseded'));

    await waitFor(() => {
      const last = urls.filter((u) => u.startsWith('GET /api/recommendations') && u.includes('limit=12')).at(-1)!;
      expect(last).toContain('status=new');
      expect(last).toContain('priority=high');
      expect(last).toContain('category=sensor_check');
      expect(last).toContain('source=heuristic');
      expect(last).toContain('plantId=8');
      expect(last).toContain('includeSuperseded=true');
      expect(last).not.toContain('offset=12');
    });
    expect(await screen.findByText('Page 1 of 3')).toBeInTheDocument();
  });

  it('searches on the server (debounced), counts the total and returns to the first page', async () => {
    const recs = recommendationsFixture();
    const { urls } = stubBackend({
      list: (p) => (p.get('q') === 'water' ? { body: [recs[0]], headers: { 'X-Total-Count': '1' } } : { body: recs, headers: { 'X-Total-Count': '30' } }),
    });
    renderPage();
    await screen.findByText('Page 1 of 3');
    fireEvent.click(screen.getByRole('button', { name: /Next/ }));
    expect(await screen.findByText('Page 2 of 3')).toBeInTheDocument();
    const before = urls.length;

    const box = screen.getByLabelText('Search recommendations');
    fireEvent.change(box, { target: { value: 'w' } });
    fireEvent.change(box, { target: { value: 'water' } });
    expect(urls.length).toBe(before); // debounced: nothing is requested while typing
    await waitFor(() => expect(urls.some((u) => u.includes('q=water'))).toBe(true));
    // Only one request for the final text, starting at the first page.
    const searches = urls.filter((u) => u.includes('q='));
    expect(searches).toHaveLength(1);
    expect(searches[0]).not.toContain('offset=12');
    // The total comes from the server's count, not from filtering the current page.
    await waitFor(() => expect(screen.getByTestId('range-label')).toHaveTextContent('Showing 1-1 of 1 directives'));
    expect(screen.queryByText(/this page/i)).not.toBeInTheDocument();
  });

  it('shows the Run empty state when there are no recommendations yet', async () => {
    stubBackend({ list: () => ({ body: [], headers: { 'X-Total-Count': '0' } }) });
    renderPage();
    expect(await screen.findByTestId('run-empty-state')).toHaveTextContent('Press Run to analyze the plant data.');
  });

  it('changes status optimistically', async () => {
    let release: (r: MockRoute) => void = () => undefined;
    const pending = new Promise<MockRoute>((resolve) => {
      release = resolve;
    });
    const recs = recommendationsFixture();
    const target = recs.find((r) => r.id === 'rec-4')!;
    const { patches } = stubBackend({
      list: () => ({ body: recs, headers: { 'X-Total-Count': '2' } }),
      patch: () => pending,
    });
    renderPage();
    const select = (await screen.findByLabelText(`Status of ${target.title}`)) as HTMLSelectElement;
    expect(select.value).toBe('new');
    fireEvent.change(select, { target: { value: 'in_progress' } });
    // Updated before the server answers.
    await waitFor(() => expect((screen.getByLabelText(`Status of ${target.title}`) as HTMLSelectElement).value).toBe('in_progress'));
    expect(patches).toEqual([{ id: 'rec-4', status: 'in_progress' }]);
    release({ body: { ...target, status: 'in_progress' } as Recommendation });
    await waitFor(() => expect(screen.queryByTestId('mutation-error')).not.toBeInTheDocument());
  });

  it('rolls back and names the blocking recommendation on a 409 conflict', async () => {
    const recs = recommendationsFixture();
    const target = recs.find((r) => r.id === 'rec-4')!;
    stubBackend({
      list: () => ({ body: recs, headers: { 'X-Total-Count': '2' } }),
      patch: () => ({
        status: 409,
        body: errorEnvelope('conflict', 'Another active recommendation exists', { activeId: 'rec-7' }),
      }),
    });
    renderPage();
    const select = (await screen.findByLabelText(`Status of ${target.title}`)) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'completed' } });

    const alert = await screen.findByTestId('mutation-error');
    expect(alert).toHaveTextContent('Cannot reopen this recommendation');
    expect(alert).toHaveTextContent('rec-7 ("Monitor and inspect", Pot 16: Krishna Tulsi (Climate Trial))');
    expect(alert).toHaveTextContent('Complete or dismiss that one first.');
    // Rolled back to the server's status.
    await waitFor(() => expect((screen.getByLabelText(`Status of ${target.title}`) as HTMLSelectElement).value).toBe('new'));
  });
});
