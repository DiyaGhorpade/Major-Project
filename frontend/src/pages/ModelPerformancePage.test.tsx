import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { ModelPerformancePage } from './ModelPerformancePage';
import { mockFetch } from '../test/helpers';
import { metricsFixture } from '../test/fixtureHelpers';

afterEach(() => vi.unstubAllGlobals());

describe('ModelPerformancePage', () => {
  it('renders real metrics without provenance or synthetic-label banners', async () => {
    mockFetch({ 'GET /api/models/performance': { body: metricsFixture() } });
    render(<ModelPerformancePage />);

    expect(await screen.findByText('xgboost_stress_classifier')).toBeInTheDocument();
    expect(screen.queryByTestId('synthetic-metrics-banner')).not.toBeInTheDocument();
    expect(screen.queryByTestId('provenance-badge')).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/synthetic|not validated/i);
    expect(screen.getAllByText(/chronological hold-out, last 25%/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Evaluated: .*IST/)).toBeInTheDocument();

    // Overall, balanced accuracy and weighted F1.
    const card = (title: string) => screen.getByText(title, { selector: 'span' }).parentElement!.parentElement!;
    expect(card('Accuracy')).toHaveTextContent('64.6%');
    expect(card('Macro Precision')).toHaveTextContent('58.3%');
    expect(card('Macro Recall')).toHaveTextContent('60.6%');
    expect(card('Macro F1')).toHaveTextContent('58.6%');
    expect(card('Balanced Accuracy')).toHaveTextContent('60.6%');
    expect(card('Weighted F1')).toHaveTextContent('64.4%');

    // Per-class table with layman status.
    expect(screen.getByText('Recognised fairly reliably on the test set')).toBeInTheDocument();
    expect(screen.getByText('563')).toBeInTheDocument();

    // Confusion matrix cells are the stored counts.
    expect(screen.getByTestId('cm-0-0')).toHaveTextContent('235');
    expect(screen.getByTestId('cm-3-3')).toHaveTextContent('419');
  });

  it('class count is data-driven (4 here, never a hard-coded 5)', async () => {
    mockFetch({ 'GET /api/models/performance': { body: metricsFixture() } });
    const { container } = render(<ModelPerformancePage />);
    await screen.findByText('xgboost_stress_classifier');
    expect(screen.getByText('4 classes')).toBeInTheDocument();
    expect(screen.getByText('Hold-out Confusion Matrix (4 classes)')).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/5 Diagnostic Classes|5 stress classes|High Confidence/i);

    // A 3-class payload adapts.
    const three = metricsFixture();
    three.confusion_matrix = { labels: ['healthy', 'low', 'high'], matrix: [[5, 1, 0], [2, 4, 1], [0, 1, 6]] };
    three.per_class = three.per_class.filter((c) => c.class_name !== 'moderate');
    vi.unstubAllGlobals();
    mockFetch({ 'GET /api/models/performance': { body: three } });
    const second = render(<ModelPerformancePage />);
    expect(await within(second.container).findByText('3 classes')).toBeInTheDocument();
  });

  it('shows baselines, anomaly model and environmental states', async () => {
    mockFetch({ 'GET /api/models/performance': { body: metricsFixture() } });
    render(<ModelPerformancePage />);
    const baselines = await screen.findByTestId('baselines');
    expect(within(baselines).getByText('Majority-class guess')).toBeInTheDocument();
    expect(within(baselines).getByText('14.1%')).toBeInTheDocument();
    expect(within(baselines).getByText('Logistic regression')).toBeInTheDocument();
    expect(within(baselines).getByText('46.1%')).toBeInTheDocument();

    const anomaly = screen.getByTestId('anomaly-model');
    expect(anomaly).toHaveTextContent('Isolation Forest');
    expect(anomaly).toHaveTextContent('anomaly_score > 0');
    expect(anomaly).toHaveTextContent('21 / 14 of 1344 rows');

    const clustering = screen.getByTestId('clustering-model');
    expect(clustering).toHaveTextContent('Number of environmental state groups');
    expect(clustering).toHaveTextContent('0.263');
    expect(clustering).not.toHaveTextContent(/healthy|synthetic/i);
    expect(anomaly).not.toHaveTextContent(/synthetic|NOT a plant-health/i);
  });
});
