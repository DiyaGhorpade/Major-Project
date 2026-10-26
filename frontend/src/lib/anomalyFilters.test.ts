import { describe, expect, it } from 'vitest';
import { parseAnomalyPlantId, parseMinScore } from './anomalyFilters';

describe('parseAnomalyPlantId', () => {
  it('accepts the integers 1-16', () => {
    expect(parseAnomalyPlantId('1')).toBe('1');
    expect(parseAnomalyPlantId('16')).toBe('16');
    expect(parseAnomalyPlantId(7)).toBe('7');
  });

  it('ignores anything else', () => {
    for (const bad of ['', '0', '17', '-3', '2.5', 'abc', '1e1', '3x', ' ', null, undefined, 0, 99]) {
      expect(parseAnomalyPlantId(bad)).toBeNull();
    }
  });
});

describe('parseMinScore', () => {
  it('passes values in range and clamps the rest to [-10, 10]', () => {
    expect(parseMinScore('0.05')).toBe(0.05);
    expect(parseMinScore('-0.2')).toBe(-0.2);
    expect(parseMinScore('50')).toBe(10);
    expect(parseMinScore('-1e9')).toBe(-10);
  });

  it('is undefined for empty or non-numeric text', () => {
    for (const bad of ['', '  ', 'abc', 'Infinity', 'NaN']) expect(parseMinScore(bad)).toBeUndefined();
  });
});
