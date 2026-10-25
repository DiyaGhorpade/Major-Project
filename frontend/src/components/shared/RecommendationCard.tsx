import { DISCLAIMERS } from './disclaimerText';
import React from 'react';
import { cn } from '../../lib/utils';
import {
  Recommendation,
  UserRecommendationStatus,
} from '../../types';
import { StatusBadge } from './StatusBadge';
import { formatDateTime } from '../../lib/utils';
import {
  Droplets,
  Sun,
  Wrench,
  CheckCircle,
  Sparkles,
  FlaskConical,
} from 'lucide-react';

type SensorStatus = Recommendation['supporting_sensor_values'][number]['status'];

const SENSOR_STATUS_STYLE: Record<SensorStatus, { label: string; chip: string; value: string }> = {
  good: { label: 'In range', chip: 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]', value: 'text-[#1C2B23]' },
  low: { label: 'Low', chip: 'bg-blue-50 text-blue-700 border-blue-200', value: 'text-blue-800' },
  high: { label: 'High', chip: 'bg-red-50 text-red-700 border-red-200', value: 'text-red-800' },
  attention: { label: 'Check', chip: 'bg-amber-50 text-amber-800 border-amber-200', value: 'text-amber-900' },
};

interface RecommendationCardProps {
  recommendation: Recommendation;
  onUpdateStatus: (id: string, status: UserRecommendationStatus) => void;
  onViewPlant?: (plantId: number) => void;
  /** True while a status change for this card is being saved (disables the controls). */
  pending?: boolean;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendation,
  onUpdateStatus,
  onViewPlant,
  pending = false,
}) => {
  const getCategoryConfig = () => {
    switch (recommendation.category) {
      case 'irrigation':
        return {
          icon: <Droplets className="w-5 h-5 text-blue-600" />,
          label: 'Watering Action',
          border: 'border-blue-200',
          bg: 'bg-blue-50/50',
          badge: 'bg-blue-100 text-blue-800',
        };
      case 'lighting':
        return {
          icon: <Sun className="w-5 h-5 text-yellow-600" />,
          label: 'Lighting Adjustment',
          border: 'border-yellow-200',
          bg: 'bg-yellow-50/50',
          badge: 'bg-yellow-100 text-yellow-800',
        };
      case 'sensor_check':
        return {
          icon: <Wrench className="w-5 h-5 text-amber-600" />,
          label: 'Sensor check',
          border: 'border-amber-200',
          bg: 'bg-amber-50/50',
          badge: 'bg-amber-100 text-amber-800',
        };
      case 'monitoring':
        return {
          icon: <Sparkles className="w-5 h-5 text-emerald-600" />,
          label: 'Monitoring',
          border: 'border-[#DCE4D9]',
          bg: 'bg-white',
          badge: 'bg-emerald-100 text-emerald-800',
        };
      default:
        return {
          icon: <Sparkles className="w-5 h-5 text-emerald-600" />,
          label: 'Care action',
          border: 'border-[#DCE4D9]',
          bg: 'bg-white',
          badge: 'bg-[#EDF4EB] text-[#2E8B57]',
        };
    }
  };

  const cat = getCategoryConfig();

  return (
    <div
      className={`rounded-card border ${cat.border} p-5 shadow-card hover:shadow-card-hover transition-all flex flex-col justify-between bg-white`}
    >
      <div>
        {/* Header Badges */}
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <StatusBadge status={recommendation.priority} size="sm" />
            <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${cat.badge}`}>
              {cat.label}
            </span>
            <button
              onClick={() => onViewPlant && onViewPlant(recommendation.plant_id)}
              className="text-xs font-bold text-[#2E8B57] hover:underline"
            >
              Pot #{recommendation.plant_id} ({recommendation.cluster})
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[#6B786E]">Status:</span>
            <select
              value={recommendation.status}
              disabled={pending}
              aria-label={`Status of ${recommendation.title}`}
              onChange={(e) => onUpdateStatus(recommendation.id, e.target.value as UserRecommendationStatus)}
              className="text-xs font-semibold bg-[#EDF4EB] text-[#173F32] border border-[#DDE8D8] rounded-md px-2 py-1 outline-none cursor-pointer focus:ring-1 focus:ring-[#2E8B57]"
            >
              <option value="new">New</option>
              <option value="in_progress">In Progress</option>
              <option value="completed">Completed</option>
              <option value="dismissed">Dismissed</option>
              {recommendation.status === 'superseded' && (
                <option value="superseded" disabled>
                  Superseded
                </option>
              )}
            </select>
          </div>
        </div>

        {/* Source and model probability */}
        <div className="flex flex-wrap items-center gap-2 mb-3 text-[11px]">
          <span
            data-testid="rec-source"
            className={cn(
              'font-semibold px-2 py-0.5 rounded-full border',
              recommendation.source === 'ml' ? 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]' : 'bg-stone-100 text-stone-700 border-stone-200',
            )}
          >
            {recommendation.source === 'ml' ? 'ML' : 'Backend rule'}
          </span>
          {recommendation.confidence !== null && recommendation.confidence !== undefined && (
            <span data-testid="rec-probability" className="font-semibold text-[#173F32]">
              Model probability {(recommendation.confidence * 100).toFixed(0)}%
            </span>
          )}
          {recommendation.status === 'superseded' && <StatusBadge status="superseded" size="sm" />}
        </div>

        {/* Title & Icon */}
        <div className="flex items-start gap-3 mb-3">
          <div className="p-2.5 rounded-xl bg-[#EDF4EB] shrink-0 mt-0.5">
            {cat.icon}
          </div>
          <div>
            <h3 className="text-base font-bold text-[#1C2B23] leading-snug">
              {recommendation.title}
            </h3>
            <span className="text-[11px] text-[#6B786E]">
              Created {formatDateTime(recommendation.created_at)} IST
            </span>
          </div>
        </div>

        {/* Simple Layman Action Callout Box (What to do!) */}
        <div className="mb-3.5 p-3.5 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8]">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[#2E8B57] block mb-1">
            Recommended Action (What to do)
          </span>
          <p className="text-xs font-semibold text-[#102F26] leading-relaxed">
            {recommendation.simpleAction}
          </p>
        </div>

        {/* Simple & Technical Explanation */}
        <div className="space-y-2 mb-4 text-xs">
          <p className="text-[#1C2B23] leading-relaxed">
            <strong>Why: </strong>{recommendation.reason}
          </p>
          <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-200/80 text-stone-700 text-[11px] leading-relaxed">
            <span className="font-semibold text-stone-900 flex items-center gap-1 mb-0.5">
              <FlaskConical className="w-3.5 h-3.5" aria-hidden="true" /> Scientific context:
            </span>
            {recommendation.scientificContext}
          </div>
        </div>

        {/* Supporting Measured Sensor Values */}
        {recommendation.supporting_sensor_values && recommendation.supporting_sensor_values.length > 0 && (
          <div className="mb-4">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[#6B786E] block mb-1.5">
              Current Sensor Evidence vs Ideal Target
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {recommendation.supporting_sensor_values.map((item, idx) => {
                const st = SENSOR_STATUS_STYLE[item.status] ?? SENSOR_STATUS_STYLE.attention;
                return (
                  <div
                    key={idx}
                    data-testid="rec-sensor-value"
                    data-status={item.status}
                    className="p-2 rounded-lg bg-white border border-[#DCE4D9] flex items-center justify-between gap-2"
                  >
                    <div>
                      <span className="text-[#6B786E] text-[11px] block">{item.metric}</span>
                      <span className={cn('font-bold', st.value)}>{item.value}</span>
                    </div>
                    <div className="text-right">
                      <span className={cn('text-[10px] font-semibold px-1.5 py-0.5 rounded border block mb-0.5', st.chip)}>{st.label}</span>
                      <span className="text-[10px] font-medium text-stone-500">Target: {item.target}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="pt-3 border-t border-[#DCE4D9] flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#6B786E]">
        <div className="space-y-0.5">
          {recommendation.basedOn && (
            <span className="block" data-testid="rec-based-on">
              Based on reading {formatDateTime(recommendation.basedOn.timestamp)} IST
              {recommendation.basedOn.modelVersion ? ` · model ${recommendation.basedOn.modelVersion}` : ''}
            </span>
          )}
          <span className="block italic">{recommendation.guidance_source}</span>
          <span className="block italic font-semibold" data-testid="rec-disclaimer">
            {DISCLAIMERS.recommendation}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {recommendation.status !== 'completed' && (
            <button
              disabled={pending}
              onClick={() => onUpdateStatus(recommendation.id, 'completed')}
              className="px-3 py-1 rounded-lg bg-[#2E8B57] text-white hover:bg-[#236e44] disabled:opacity-60 font-bold text-xs transition-colors flex items-center gap-1 shadow-sm"
            >
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Mark Done</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
