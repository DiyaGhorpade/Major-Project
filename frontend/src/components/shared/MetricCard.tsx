import React, { ReactNode } from 'react';
import { cn } from '../../lib/utils';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  trend?: {
    value: string;
    direction: 'up' | 'down' | 'neutral';
    label?: string;
  };
  badge?: ReactNode;
  className?: string;
  onClick?: () => void;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  badge,
  className,
  onClick,
}) => {
  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card transition-all duration-200',
        onClick && 'cursor-pointer hover:shadow-card-hover hover:border-[#426B4F]',
        className
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-[#6B786E]">
          {title}
        </span>
        {icon && (
          <div className="w-8 h-8 rounded-lg bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center">
            {icon}
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2 mb-1">
        <div className="text-2xl font-bold tracking-tight text-[#1C2B23]">
          {value}
        </div>
        {badge}
      </div>

      <div className="flex items-center justify-between text-xs text-[#6B786E]">
        {subtitle && <span>{subtitle}</span>}
        {trend && (
          <div
            className={cn(
              'inline-flex items-center gap-0.5 font-medium',
              trend.direction === 'up' && 'text-[#2E8B57]',
              trend.direction === 'down' && 'text-[#C94C4C]',
              trend.direction === 'neutral' && 'text-[#6B786E]'
            )}
          >
            {trend.direction === 'up' && <ArrowUpRight className="w-3.5 h-3.5" />}
            {trend.direction === 'down' && <ArrowDownRight className="w-3.5 h-3.5" />}
            <span>{trend.value}</span>
            {trend.label && <span className="text-stone-400 ml-1">({trend.label})</span>}
          </div>
        )}
      </div>
    </div>
  );
};
