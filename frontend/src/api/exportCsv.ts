import { SensorRecord } from '../types';

export function exportRecordsToCsv(records: SensorRecord[], filenamePrefix = 'plant_monitor_telemetry'): void {
  if (!records || records.length === 0) {
    alert('No records available to export for current filters.');
    return;
  }

  // Exact CSV headers matching specified schema
  const headers = [
    'timestamp',
    'session_id',
    'node_id',
    'sampling_point',
    'plant_id',
    'soil',
    'temperature',
    'humidity',
    'light',
  ];

  const rows = records.map((r) => [
    `"${r.timestamp}"`,
    r.session_id,
    `"${r.node_id}"`,
    r.sampling_point,
    r.plant_id,
    typeof r.soil === 'number' ? r.soil.toFixed(1) : '',
    typeof r.temperature === 'number' ? r.temperature.toFixed(1) : '',
    typeof r.humidity === 'number' ? r.humidity.toFixed(1) : '',
    typeof r.light === 'number' ? r.light.toFixed(1) : '',
  ]);

  const csvContent = [headers.join(','), ...rows.map((row) => row.join(','))].join('\r\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  link.setAttribute('href', url);
  link.setAttribute('download', `${filenamePrefix}_${timestamp}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
