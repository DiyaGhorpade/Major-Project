import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { RecommendationCard } from './RecommendationCard';
import { recommendationsFixture } from '../../test/fixtureHelpers';

const irrigation = () => recommendationsFixture().find((r) => r.category === 'irrigation')!;

describe('RecommendationCard', () => {
  it('shows priority, source, model probability, reason and basedOn, with no advisory disclaimer', () => {
    const rec = irrigation();
    render(<RecommendationCard recommendation={rec} onUpdateStatus={vi.fn()} />);
    expect(screen.getByText('High Priority')).toBeInTheDocument();
    expect(screen.getByTestId('rec-source')).toHaveTextContent('ML');
    expect(screen.getByTestId('rec-probability')).toHaveTextContent('Model probability 57%');
    expect(screen.getByText(/Soil moisture 2\.5 % \(recent average 2\.1 %\); model P\(high\)=0\.57\./)).toBeInTheDocument();
    expect(screen.queryByTestId('rec-disclaimer')).not.toBeInTheDocument();
    expect(screen.getByTestId('rec-based-on')).toHaveTextContent(/Based on reading Oct 18, 22:02:15 IST · model 1\.0\.0/);
    expect(screen.getByText(/ML rules · model 1\.0\.0/)).toBeInTheDocument();
    expect(screen.getByText(/Tulsi \(Ocimum\) generally prefers evenly moist/)).toBeInTheDocument();
  });

  it('colours supporting sensor values by status', () => {
    render(<RecommendationCard recommendation={irrigation()} onUpdateStatus={vi.fn()} />);
    const rows = screen.getAllByTestId('rec-sensor-value');
    expect(rows).toHaveLength(4);
    expect(rows[0]).toHaveAttribute('data-status', 'low');
    expect(within(rows[0]).getByText('Low')).toHaveClass('bg-blue-50');
    expect(rows[1]).toHaveAttribute('data-status', 'good');
    expect(within(rows[1]).getByText('In range')).toHaveClass('bg-[#EDF4EB]');
  });

  it('labels monitoring and sensor_check categories and a backend-rule source without a probability', () => {
    const base = recommendationsFixture().find((r) => r.category === 'monitoring')!;
    const { rerender } = render(<RecommendationCard recommendation={base} onUpdateStatus={vi.fn()} />);
    expect(screen.getByText('Monitoring')).toBeInTheDocument();

    rerender(
      <RecommendationCard
        recommendation={{ ...base, category: 'sensor_check', source: 'heuristic', confidence: null }}
        onUpdateStatus={vi.fn()}
      />,
    );
    expect(screen.getByText('Sensor check')).toBeInTheDocument();
    expect(screen.getByTestId('rec-source')).toHaveTextContent('Backend rule');
    expect(screen.queryByTestId('rec-probability')).not.toBeInTheDocument();
  });

  it('keeps the original props working: status select and Mark Done', () => {
    const onUpdateStatus = vi.fn();
    const onViewPlant = vi.fn();
    const rec = irrigation();
    render(<RecommendationCard recommendation={rec} onUpdateStatus={onUpdateStatus} onViewPlant={onViewPlant} />);
    fireEvent.click(screen.getByRole('button', { name: /Mark Done/ }));
    expect(onUpdateStatus).toHaveBeenCalledWith(rec.id, 'completed');
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'in_progress' } });
    expect(onUpdateStatus).toHaveBeenCalledWith(rec.id, 'in_progress');
    fireEvent.click(screen.getByRole('button', { name: /Pot #8/ }));
    expect(onViewPlant).toHaveBeenCalledWith(8);
  });
});
