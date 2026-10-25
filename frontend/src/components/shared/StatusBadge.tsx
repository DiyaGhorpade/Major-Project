import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  HelpCircle,
  Wifi,
  WifiOff,
  Activity,
  Droplets,
  Sun,
  Sprout,
  Thermometer,
} from 'lucide-react';
import { cn } from '../../lib/utils';

export type BadgeVariant =
  | 'online'
  | 'offline'
  | 'unknown'
  | 'healthy'
  | 'attention'
  | 'critical'
  | 'stale'
  | 'unmeasured'
  | 'accepted'
  | 'rejected'
  | 'flagged'
  | 'completed'
  | 'in_progress'
  | 'scheduled'
  | 'high'
  | 'medium'
  | 'low'
  | 'water stress'
  | 'light stress'
  | 'nutrient stress'
  | 'temperature/humidity stress'
  | 'mixed stress'
  | 'unclassified stress'
  | 'failed'
  | 'partial'
  | 'waiting'
  | 'superseded'
  | 'queued'
  | 'running';

interface StatusBadgeProps {
  status: BadgeVariant | string;
  label?: string;
  size?: 'sm' | 'md';
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  size = 'md',
  className,
}) => {
  const norm = status.toLowerCase();

  let text = label;
  let bg = 'bg-stone-100 text-stone-700 border-stone-200';
  let icon = <HelpCircle className="w-3.5 h-3.5" />;

  switch (norm) {
    case 'online':
      text = text || 'Online';
      bg = 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]';
      icon = <Wifi className="w-3.5 h-3.5" />;
      break;
    case 'offline':
      text = text || 'Offline';
      bg = 'bg-red-50 text-red-700 border-red-200';
      icon = <WifiOff className="w-3.5 h-3.5" />;
      break;
    case 'unknown':
      text = text || 'Unknown';
      bg = 'bg-stone-100 text-stone-600 border-stone-200';
      icon = <HelpCircle className="w-3.5 h-3.5" />;
      break;
    case 'unmeasured':
      text = text || 'Not analyzed yet';
      bg = 'bg-stone-100 text-stone-600 border-stone-200';
      icon = <Clock className="w-3.5 h-3.5" />;
      break;
    case 'failed':
      text = text || 'Failed';
      bg = 'bg-red-50 text-red-700 border-red-200';
      icon = <XCircle className="w-3.5 h-3.5" />;
      break;
    case 'partial':
      text = text || 'Partial';
      bg = 'bg-amber-50 text-amber-800 border-amber-200';
      icon = <AlertTriangle className="w-3.5 h-3.5" />;
      break;
    case 'scheduled':
      text = text || 'Scheduled';
      bg = 'bg-blue-50 text-blue-700 border-blue-200';
      icon = <Clock className="w-3.5 h-3.5" />;
      break;
    case 'waiting':
      text = text || 'Waiting';
      bg = 'bg-stone-100 text-stone-600 border-stone-200';
      icon = <Clock className="w-3.5 h-3.5" />;
      break;
    case 'superseded':
      text = text || 'Superseded';
      bg = 'bg-stone-100 text-stone-500 border-stone-200';
      icon = <Clock className="w-3.5 h-3.5" />;
      break;
    case 'queued':
      text = text || 'Queued';
      bg = 'bg-blue-50 text-blue-700 border-blue-200';
      icon = <Clock className="w-3.5 h-3.5" />;
      break;
    case 'running':
      text = text || 'Running';
      bg = 'bg-blue-50 text-blue-700 border-blue-200';
      icon = <Activity className="w-3.5 h-3.5 animate-pulse" />;
      break;
    case 'healthy':
    case 'optimal':
      text = text || 'Healthy (Control)';
      bg = 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]';
      icon = <CheckCircle2 className="w-3.5 h-3.5" />;
      break;
    case 'water stress':
      text = text || 'Water Stress';
      bg = 'bg-amber-50 text-amber-900 border-amber-300';
      icon = <Droplets className="w-3.5 h-3.5 text-amber-600" />;
      break;
    case 'light stress':
      text = text || 'Light Stress';
      bg = 'bg-yellow-50 text-yellow-900 border-yellow-300';
      icon = <Sun className="w-3.5 h-3.5 text-yellow-600" />;
      break;
    case 'nutrient stress':
      text = text || 'Nutrient Stress';
      bg = 'bg-purple-50 text-purple-900 border-purple-300';
      icon = <Sprout className="w-3.5 h-3.5 text-purple-600" />;
      break;
    case 'temperature/humidity stress':
      text = text || 'Temperature/Humidity Stress';
      bg = 'bg-sky-50 text-sky-900 border-sky-300';
      icon = <Thermometer className="w-3.5 h-3.5 text-sky-600" />;
      break;
    case 'mixed stress':
      text = text || 'Mixed Stress';
      bg = 'bg-orange-50 text-orange-900 border-orange-300';
      icon = <AlertTriangle className="w-3.5 h-3.5 text-orange-600" />;
      break;
    case 'unclassified stress':
      text = text || 'Unclassified Stress';
      bg = 'bg-stone-100 text-stone-700 border-stone-200';
      icon = <HelpCircle className="w-3.5 h-3.5" />;
      break;
    case 'attention':
    case 'warning':
      text = text || 'Stress Detected';
      bg = 'bg-amber-50 text-amber-800 border-amber-200';
      icon = <AlertTriangle className="w-3.5 h-3.5" />;
      break;
    case 'critical':
      text = text || 'Severe Stress';
      bg = 'bg-red-50 text-red-700 border-red-200';
      icon = <XCircle className="w-3.5 h-3.5" />;
      break;
    case 'stale':
      text = text || 'Stale Reading';
      bg = 'bg-stone-100 text-stone-600 border-stone-200';
      icon = <Clock className="w-3.5 h-3.5" />;
      break;
    case 'accepted':
      text = text || 'Accepted';
      bg = 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]';
      icon = <CheckCircle2 className="w-3.5 h-3.5" />;
      break;
    case 'rejected':
      text = text || 'Rejected';
      bg = 'bg-red-50 text-red-700 border-red-200';
      icon = <XCircle className="w-3.5 h-3.5" />;
      break;
    case 'flagged':
      text = text || 'Flagged';
      bg = 'bg-amber-50 text-amber-800 border-amber-200';
      icon = <AlertTriangle className="w-3.5 h-3.5" />;
      break;
    case 'completed':
      text = text || 'Completed';
      bg = 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]';
      icon = <CheckCircle2 className="w-3.5 h-3.5" />;
      break;
    case 'in_progress':
      text = text || 'In Progress';
      bg = 'bg-blue-50 text-blue-700 border-blue-200';
      icon = <Clock className="w-3.5 h-3.5 animate-spin" />;
      break;
    case 'high':
      text = text || 'High Priority';
      bg = 'bg-red-50 text-red-700 border-red-200 font-medium';
      icon = <AlertTriangle className="w-3.5 h-3.5" />;
      break;
    case 'medium':
      text = text || 'Medium Priority';
      bg = 'bg-amber-50 text-amber-800 border-amber-200';
      icon = <AlertTriangle className="w-3.5 h-3.5" />;
      break;
    case 'low':
      text = text || 'Low Priority';
      bg = 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]';
      icon = <CheckCircle2 className="w-3.5 h-3.5" />;
      break;
    default:
      text = text || status;
      bg = 'bg-stone-100 text-stone-700 border-stone-200';
      icon = <Activity className="w-3.5 h-3.5" />;
      break;
  }

  const sizeClasses =
    size === 'sm'
      ? 'px-2 py-0.5 text-xs gap-1.5'
      : 'px-2.5 py-1 text-xs gap-1.5';

  return (
    <span
      className={cn(
        'inline-flex items-center font-medium rounded-full border leading-none select-none',
        sizeClasses,
        bg,
        className
      )}
    >
      {icon}
      <span>{text}</span>
    </span>
  );
};
