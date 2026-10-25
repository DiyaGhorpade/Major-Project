import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SensorReadingCard } from './SensorReadingCard';
import { isNightReading } from '../../lib/utils';

const LIGHT = { min: 300, max: 20000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Bright', highLabel: 'Glare', daytimeHours: [9, 16] as [number, number] };

describe('isNightReading', () => {
  it('uses the IST hour against the inclusive daytime hours', () => {
    expect(isNightReading('2026-10-18T22:00:45+05:30', [9, 16])).toBe(true);
    expect(isNightReading('2026-10-18T08:59:00+05:30', [9, 16])).toBe(true);
    expect(isNightReading('2026-10-18T09:00:00+05:30', [9, 16])).toBe(false);
    expect(isNightReading('2026-10-18T16:59:00+05:30', [9, 16])).toBe(false);
    expect(isNightReading('2026-10-18T17:00:00+05:30', [9, 16])).toBe(true);
    expect(isNightReading('2026-10-18T04:00:00+00:00', [9, 16])).toBe(false); // 09:30 IST
    expect(isNightReading('2026-10-18T12:00:00+00:00', [9, 16])).toBe(true); // 17:30 IST
  });

  it('guesses nothing without hours or a parseable time', () => {
    expect(isNightReading('2026-10-18T22:00:00+05:30', undefined)).toBe(false);
    expect(isNightReading(undefined, [9, 16])).toBe(false);
    expect(isNightReading('not a date', [9, 16])).toBe(false);
  });
});

describe('SensorReadingCard light at night', () => {
  it('shows a neutral "Night" state instead of "Shaded / Dim"', () => {
    render(<SensorReadingCard type="light" value={0} optimalRange={LIGHT} readingTime="2026-10-18T22:00:45+05:30" timestamp="Oct 18, 22:00:45" />);
    expect(screen.getByText('Night')).toBeInTheDocument();
    expect(screen.queryByText('Shaded / Dim')).not.toBeInTheDocument();
    expect(screen.getByText(/No light target at night/)).toBeInTheDocument();
  });

  it('still warns about dim light in the daytime', () => {
    render(<SensorReadingCard type="light" value={50} optimalRange={LIGHT} readingTime="2026-10-18T12:00:00+05:30" />);
    expect(screen.getByText('Shaded / Dim')).toBeInTheDocument();
    expect(screen.queryByText('Night')).not.toBeInTheDocument();
  });

  it('does not change other sensors at night', () => {
    render(
      <SensorReadingCard
        type="soil"
        value={10}
        optimalRange={{ min: 25, max: 75, unit: '%' }}
        readingTime="2026-10-18T22:00:45+05:30"
      />,
    );
    expect(screen.getByText('Thirsty / Dry')).toBeInTheDocument();
  });
});
