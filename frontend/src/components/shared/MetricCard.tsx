import React, { ReactNode } from 'react';
import { cn } from '../../lib/utils';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  badge?: ReactNode;
  className?: string;
  /** Makes the card a keyboard-accessible button. */
  onClick?: () => void;
}

export const MetricCard: React.FC<MetricCardProps> = ({ title, value, subtitle, icon, badge, className, onClick }) => {
  return (
    <div
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      className={cn(
        'bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card transition-all duration-200',
        onClick && 'cursor-pointer hover:shadow-card-hover hover:border-[#426B4F] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2E8B57]',
        className
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-[#6B786E]">{title}</span>
        {icon && <div className="w-8 h-8 rounded-lg bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center">{icon}</div>}
      </div>

      <div className="flex items-baseline gap-2 mb-1">
        <div className="text-2xl font-bold tracking-tight text-[#1C2B23]">{value}</div>
        {badge}
      </div>

      {subtitle && (
        <div className="flex items-center justify-between text-xs text-[#6B786E]">
          <span>{subtitle}</span>
        </div>
      )}
    </div>
  );
};
