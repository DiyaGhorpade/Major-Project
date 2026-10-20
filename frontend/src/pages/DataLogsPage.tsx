import React, { useState, useEffect } from 'react';
import {
  Download,
  Search,
  ChevronLeft,
  ChevronRight,
  Database,
  Filter,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { SensorRecord, Plant } from '../types';
import { StatusBadge } from '../components/shared/StatusBadge';
import { RecordDetailDrawer } from '../components/shared/RecordDetailDrawer';
import { exportRecordsToCsv } from '../api/exportCsv';
import { formatDateTime } from '../lib/utils';

export const DataLogsPage: React.FC = () => {
  const [records, setRecords] = useState<SensorRecord[]>([]);
  const [plants, setPlants] = useState<Plant[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<SensorRecord | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [plantFilter, setPlantFilter] = useState('all');
  const [nodeFilter, setNodeFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(15);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [recs, pls] = await Promise.all([
        apiClient.getRecords(),
        apiClient.getPlants(),
      ]);
      setRecords(recs);
      setPlants(pls);
    } finally {
      setLoading(false);
    }
  };

  const filteredRecords = records.filter((r) => {
    if (plantFilter !== 'all' && r.plant_id !== Number(plantFilter)) return false;
    if (nodeFilter !== 'all' && r.node_id !== nodeFilter) return false;
    if (statusFilter !== 'all' && r.status !== statusFilter) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchPlant = `pot ${r.plant_id}`.includes(q) || String(r.plant_id).includes(q);
      const matchSession = `session ${r.session_id}`.includes(q) || String(r.session_id).includes(q);
      const matchNode = `node ${r.node_id}`.includes(q);
      const matchPoint = `point ${r.sampling_point}`.includes(q) || String(r.sampling_point).includes(q);
      const matchTs = r.timestamp.toLowerCase().includes(q);
      return matchPlant || matchSession || matchNode || matchPoint || matchTs;
    }
    return true;
  });

  const totalRecords = filteredRecords.length;
  const totalPages = Math.max(1, Math.ceil(totalRecords / pageSize));
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedRecords = filteredRecords.slice(startIndex, startIndex + pageSize);

  const handleExportCsv = () => {
    exportRecordsToCsv(filteredRecords, 'tulsi_telemetry_dataset');
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Export Action */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              Raw Telemetry Repository
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              {totalRecords} Logged Records
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            Historical Sensor Telemetry Database
          </h1>
          <p className="text-xs text-[#6B786E] mt-1">
            Exact physical telemetry records from Node 1 and Node 2. Click any row for raw JSON payload inspection.
          </p>
        </div>

        <button
          onClick={handleExportCsv}
          className="px-4 py-2 bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-bold rounded-lg flex items-center gap-2 transition-all shadow-sm"
        >
          <Download className="w-4 h-4" />
          <span>Download Filtered CSV ({filteredRecords.length})</span>
        </button>
      </div>

      {/* Filter Toolbar */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 flex-1">
          {/* Search */}
          <div className="relative min-w-[220px] flex-1 max-w-xs">
            <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by Pot #, Session, or date..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full text-xs bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg pl-9 pr-3 py-1.5 outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
            />
          </div>

          {/* Plant filter */}
          <select
            value={plantFilter}
            onChange={(e) => {
              setPlantFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Pots (1 to 16)</option>
            {plants.map((p) => (
              <option key={p.id} value={p.id}>
                Pot #{p.id} ({p.variety})
              </option>
            ))}
          </select>

          {/* Node filter */}
          <select
            value={nodeFilter}
            onChange={(e) => {
              setNodeFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Nodes</option>
            <option value="1">Node 1 (Pots 1–12)</option>
            <option value="2">Node 2 (Pots 13–16)</option>
          </select>

          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="text-xs bg-white border border-[#DCE4D9] rounded-lg px-3 py-1.5 text-[#1C2B23] outline-none shadow-sm cursor-pointer"
          >
            <option value="all">All Statuses</option>
            <option value="accepted">Accepted Only</option>
            <option value="flagged">Flagged Alerts</option>
          </select>
        </div>

        {/* Page size selector */}
        <div className="flex items-center gap-2 text-xs text-[#6B786E]">
          <span>Per page:</span>
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="bg-[#EDF4EB] border border-[#DDE8D8] rounded px-2 py-1 font-semibold text-[#173F32] outline-none cursor-pointer"
          >
            <option value={10}>10</option>
            <option value={15}>15</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
          </select>
        </div>
      </div>

      {/* Main Records Table — Exact Specified Schema Headers */}
      <div className="bg-white rounded-card border border-[#DCE4D9] shadow-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#DCE4D9] text-[#6B786E] bg-[#F6F7F0]/60">
                <th className="py-3 px-4 font-semibold">timestamp</th>
                <th className="py-3 px-4 font-semibold">session_id</th>
                <th className="py-3 px-4 font-semibold">node_id</th>
                <th className="py-3 px-4 font-semibold">sampling_point</th>
                <th className="py-3 px-4 font-semibold">plant_id</th>
                <th className="py-3 px-4 font-semibold">soil (%)</th>
                <th className="py-3 px-4 font-semibold">temperature (°C)</th>
                <th className="py-3 px-4 font-semibold">humidity (%)</th>
                <th className="py-3 px-4 font-semibold">light (lux)</th>
                <th className="py-3 px-4 font-semibold text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#DCE4D9]/60 font-mono">
              {paginatedRecords.map((r, idx) => (
                <tr
                  key={r.record_id || idx}
                  onClick={() => setSelectedRecord(r)}
                  className="hover:bg-[#EDF4EB]/40 transition-colors cursor-pointer"
                >
                  <td className="py-3 px-4 text-[#1C2B23] font-medium">
                    {r.timestamp}
                  </td>
                  <td className="py-3 px-4 text-stone-700">
                    {r.session_id}
                  </td>
                  <td className="py-3 px-4 font-bold text-stone-800">
                    "{r.node_id}"
                  </td>
                  <td className="py-3 px-4 text-stone-700">
                    {r.sampling_point}
                  </td>
                  <td className="py-3 px-4 font-bold text-[#102F26]">
                    Pot #{r.plant_id}
                  </td>
                  <td className="py-3 px-4 font-bold text-[#2E8B57]">
                    {r.soil.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-[#1C2B23] font-semibold">
                    {r.temperature.toFixed(1)}°C
                  </td>
                  <td className="py-3 px-4 text-[#1C2B23]">
                    {r.humidity.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-[#1C2B23]">
                    {Math.round(r.light).toLocaleString()} lx
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedRecord(r);
                      }}
                      className="px-2.5 py-1 text-xs rounded bg-[#EDF4EB] text-[#2E8B57] font-semibold hover:bg-[#DDE8D8] transition-colors"
                    >
                      Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 border-t border-[#DCE4D9] bg-[#F6F7F0]/40 flex flex-wrap items-center justify-between gap-3 text-xs text-[#6B786E]">
          <div>
            Showing <span className="font-semibold text-[#1C2B23]">{startIndex + 1}</span> to{' '}
            <span className="font-semibold text-[#1C2B23]">
              {Math.min(startIndex + pageSize, totalRecords)}
            </span>{' '}
            of <span className="font-semibold text-[#1C2B23]">{totalRecords}</span> records
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-md border border-[#DCE4D9] bg-white text-[#1C2B23] disabled:opacity-40 hover:bg-[#EDF4EB] transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            <span className="px-3 font-semibold text-[#1C2B23]">
              Page {currentPage} of {totalPages}
            </span>

            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-md border border-[#DCE4D9] bg-white text-[#1C2B23] disabled:opacity-40 hover:bg-[#EDF4EB] transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <RecordDetailDrawer
        record={selectedRecord}
        onClose={() => setSelectedRecord(null)}
      />
    </div>
  );
};
