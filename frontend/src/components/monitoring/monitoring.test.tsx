import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SensorTrendChart } from '../charts/SensorTrendChart';
import { LiveSparkline } from '../charts/LiveSparkline';
import { describeCadence, filterByRange, formatDuration, formatIstTick, formatPlantIds } from './format';
import { monitoringFixture } from '../../test/monitoringFixtures';

describe('range filtering (relative to the dataset latest timestamp)', () => {
  const series = monitoringFixture().series; // 30 points, 2 h apart, ending 2026-10-18T22:01:30+05:30
  it('keeps the last 24 h / 7 days / everything', () => {
    expect(filterByRange(series, '24h')).toHaveLength(13);
    expect(filterByRange(series, '7d')).toHaveLength(30);
    expect(filterByRange(series, 'all')).toHaveLength(30);
    expect(filterByRange([], '24h')).toEqual([]);
  });
});

describe('format helpers', () => {
  it('formats IST axis ticks from +05:30 timestamps', () => {
    const ms = new Date('2026-10-18T14:00:00+05:30').getTime();
    expect(formatIstTick(ms, false)).toBe('14:00');
    expect(formatIstTick(ms, true)).toBe('18 Oct 14:00');
  });
  it('describes the cadence from collection times', () => {
    expect(describeCadence(['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'])).toBe(
      'every 2 h (12 per day)',
    );
    expect(describeCadence(['08:00', '13:00', '18:00'])).toBe('3 per day (08:00, 13:00, 18:00)');
    expect(describeCadence([])).toBeNull();
  });
  it('formats durations and pot lists', () => {
    expect(formatDuration(597915)).toBe('6d 22h');
    expect(formatDuration(null)).toBe('n/a');
    expect(formatPlantIds([1, 2, 3, 4])).toBe('Pots 1–4');
    expect(formatPlantIds([2, 4, 6])).toBe('Pots 2, 4, 6');
  });
});

describe('charts', () => {
  it('SensorTrendChart tells the user when every value is missing instead of plotting zeros', () => {
    const records = [
      { timestamp: '2026-10-18T10:00:00+05:30', soil: null, temperature: null, humidity: null, light: null },
      { timestamp: '2026-10-18T12:00:00+05:30', soil: null, temperature: null, humidity: null, light: null },
    ];
    render(<SensorTrendChart records={records} />);
    expect(screen.getByTestId('trend-empty')).toHaveTextContent('No readings to chart');
  });

  it('SensorTrendChart renders with partially missing values and notes the IST axis', () => {
    const records = monitoringFixture().series.map((p, i) => (i % 5 === 0 ? { ...p, soil: null } : p));
    render(<SensorTrendChart records={records} title="Trend" />);
    expect(screen.getByText('Trend')).toBeInTheDocument();
    expect(screen.getByText(/Times shown in IST/)).toBeInTheDocument();
    expect(screen.queryByTestId('trend-empty')).not.toBeInTheDocument();
  });

  it('LiveSparkline shows n/a for an all-null series', () => {
    render(<LiveSparkline data={[null, null]} />);
    expect(screen.getByText('n/a')).toBeInTheDocument();
  });
});
