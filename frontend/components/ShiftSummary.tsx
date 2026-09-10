"use client";

// Shift summary: operator briefing dashboard showing the last shift's key
// metrics, worst behaviors, and hazard zones.

import { useEffect, useState } from "react";
import { getShiftSummary, ShiftSummary as ShiftSummaryType } from "@/lib/api";

export default function ShiftSummary() {
  const [data, setData] = useState<ShiftSummaryType | null>(null);
  const [windowDays, setWindowDays] = useState(1);

  useEffect(() => {
    getShiftSummary(windowDays).then(setData).catch(() => {});
  }, [windowDays]);

  if (!data) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold text-slate-700">Shift summary</h2>
        <p className="rounded-lg bg-slate-50 px-3 py-6 text-center text-sm text-slate-400">
          Loading shift briefing…
        </p>
      </section>
    );
  }

  const riskColor =
    data.overall_risk > 0.7
      ? "text-red-600"
      : data.overall_risk > 0.4
        ? "text-amber-600"
        : "text-emerald-600";

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Shift summary</h2>
        <div className="flex items-center gap-2">
          {[1, 7, 30].map((d) => (
            <button
              key={d}
              onClick={() => setWindowDays(d)}
              className={`rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors ${
                windowDays === d
                  ? "bg-indigo-100 text-indigo-700"
                  : "text-slate-400 hover:bg-slate-100"
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
      </header>

      {/* Narrative */}
      <p className="mb-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
        {data.narrative}
      </p>

      {/* KPI row */}
      <div className="mb-3 grid grid-cols-4 gap-2">
        <div className="rounded-lg bg-slate-50 p-2 text-center">
          <p className="text-lg font-bold text-slate-900">{data.total_events}</p>
          <p className="text-[10px] text-slate-400">total events</p>
        </div>
        <div className="rounded-lg bg-slate-50 p-2 text-center">
          <p className={`text-lg font-bold ${data.serious_count > 0 ? "text-orange-600" : "text-emerald-600"}`}>
            {data.serious_count}
          </p>
          <p className="text-[10px] text-slate-400">high/critical</p>
        </div>
        <div className="rounded-lg bg-slate-50 p-2 text-center">
          <p className={`text-lg font-bold ${riskColor}`}>
            {data.avg_risk.toFixed(2)}
          </p>
          <p className="text-[10px] text-slate-400">avg risk</p>
        </div>
        <div className="rounded-lg bg-slate-50 p-2 text-center">
          <p className="text-lg font-bold text-slate-900">{data.zones_affected}</p>
          <p className="text-[10px] text-slate-400">zones hit</p>
        </div>
      </div>

      {/* Top behaviors */}
      {data.behaviors.length > 0 && (
        <div className="mb-2">
          <p className="mb-1 text-[11px] font-medium text-slate-500">Top behaviors</p>
          <div className="space-y-1">
            {data.behaviors.slice(0, 5).map((b) => (
              <div
                key={b.behavior_class}
                className="flex items-center justify-between rounded-md bg-slate-50 px-2 py-1"
              >
                <span className="text-xs text-slate-700">
                  {b.behavior_class.replace(/_/g, " ")}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">
                    {b.n}x · avg {b.avg_risk.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Worst bays */}
      {data.top_bays.length > 0 && (
        <div>
          <p className="mb-1 text-[11px] font-medium text-slate-500">Highest-risk zones</p>
          <div className="flex flex-wrap gap-1.5">
            {data.top_bays.map((bay) => (
              <span
                key={bay.zone_id}
                className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-[10px] font-medium text-red-700"
              >
                {bay.zone_id}: {bay.max_risk.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
