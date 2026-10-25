import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, Clock, HelpCircle, Wifi, WifiOff, Activity } from 'lucide-react';
import { cn } from '../../lib/utils';
import { PLANT_STATUS_LABELS } from '../../lib/statusLabels';

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
  | 'failed'
  | 'partial'
  | 'waiting'
  | 'superseded';

interface StatusBadgeProps {
  status: BadgeVariant | string;
  label?: string;
  size?: 'sm' | 'md';
  className?: string;
}

const GREEN = 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]';
const RED = 'bg-red-50 text-red-700 border-red-200';
const AMBER = 'bg-amber-50 text-amber-800 border-amber-200';
const BLUE = 'bg-blue-50 text-blue-700 border-blue-200';
const STONE = 'bg-stone-100 text-stone-600 border-stone-200';

const OK = <CheckCircle2 className="w-3.5 h-3.5" />;
const WARN = <AlertTriangle className="w-3.5 h-3.5" />;
const CLOCK = <Clock className="w-3.5 h-3.5" />;
const BAD = <XCircle className="w-3.5 h-3.5" />;

interface Style {
  text: string;
  cls: string;
  icon: React.ReactNode;
}

/** Statuses the pages actually show: devices, readings, sessions/nodes, recommendations and plant status. */
const STYLES: Record<string, Style> = {
  online: { text: 'Online', cls: GREEN, icon: <Wifi className="w-3.5 h-3.5" /> },
  offline: { text: 'Offline', cls: RED, icon: <WifiOff className="w-3.5 h-3.5" /> },
  unknown: { text: 'Unknown', cls: STONE, icon: <HelpCircle className="w-3.5 h-3.5" /> },
  // Plant status derived from the model's severity (copy shared with the severity badges).
  healthy: { text: PLANT_STATUS_LABELS.healthy, cls: GREEN, icon: OK },
  attention: { text: PLANT_STATUS_LABELS.attention, cls: AMBER, icon: WARN },
  critical: { text: PLANT_STATUS_LABELS.critical, cls: RED, icon: BAD },
  stale: { text: PLANT_STATUS_LABELS.stale, cls: STONE, icon: CLOCK },
  unmeasured: { text: PLANT_STATUS_LABELS.unmeasured, cls: STONE, icon: CLOCK },
  // Stored readings
  accepted: { text: 'Accepted', cls: GREEN, icon: OK },
  rejected: { text: 'Rejected', cls: RED, icon: BAD },
  flagged: { text: 'Flagged', cls: AMBER, icon: WARN },
  // Sessions and per-node progress
  completed: { text: 'Completed', cls: GREEN, icon: OK },
  in_progress: { text: 'In Progress', cls: BLUE, icon: <Clock className="w-3.5 h-3.5 animate-spin" /> },
  scheduled: { text: 'Scheduled', cls: BLUE, icon: CLOCK },
  failed: { text: 'Failed', cls: RED, icon: BAD },
  partial: { text: 'Partial', cls: AMBER, icon: WARN },
  waiting: { text: 'Waiting', cls: STONE, icon: CLOCK },
  // Recommendations
  superseded: { text: 'Superseded', cls: 'bg-stone-100 text-stone-500 border-stone-200', icon: CLOCK },
  high: { text: 'High Priority', cls: `${RED} font-medium`, icon: WARN },
  medium: { text: 'Medium Priority', cls: AMBER, icon: WARN },
  low: { text: 'Low Priority', cls: GREEN, icon: OK },
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label, size = 'md', className }) => {
  const style = STYLES[status.toLowerCase()] ?? {
    text: status,
    cls: 'bg-stone-100 text-stone-700 border-stone-200',
    icon: <Activity className="w-3.5 h-3.5" />,
  };

  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs gap-1.5' : 'px-2.5 py-1 text-xs gap-1.5';

  return (
    <span className={cn('inline-flex items-center font-medium rounded-full border leading-none select-none', sizeClasses, style.cls, className)}>
      {style.icon}
      <span>{label ?? style.text}</span>
    </span>
  );
};
