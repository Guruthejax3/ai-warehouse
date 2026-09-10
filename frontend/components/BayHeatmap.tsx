"use client";

// Bay risk heatmap: each warehouse bay colored by its highest-risk event.
// Worst bay risk always wins (a single critical event colors the whole bay red).

import { useEffect, useState } from "react";
import { getEvents, RiskEvent, RISK_ORDER, RISK_STYLES } from "@/lib/api";

const BAYS = Array.from({ length: 8 }, (_, i) => `bay_${i}`);

export default function BayHeatmap() {
  const [events, setEvents] = useState<RiskEvent[]>([]);

  useEffect(() => {
    getEvents().then(setEvents).catch(() => {});
  }, []);

  const worst = new Map<string, RiskEvent>();
  for (const ev of events) {
    const cur = worst.get(ev.zone_id);
    if (!cur || RISK_ORDER[ev.risk_level] > RISK_ORDER[cur.risk_level]) {
      worst.set(ev.zone_id, ev);
    }
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-slate-700">Bay risk heatmap</h2>
      <div className="grid grid-cols-4 gap-2">
        {BAYS.map((bay) => {
          const ev = worst.get(bay);
          const level = ev?.risk_level ?? "low";
          const heat = RISK_STYLES[level].heat;
          return (
            <div
              key={bay}
              className="relative flex h-16 flex-col justify-between rounded-lg bg-slate-100 p-2 ring-1 ring-slate-200"
              style={{ background: ev ? `color-mix(in srgb, ${heat} 22%, white)` : undefined }}
            >
              <span className="text-[11px] font-semibold text-slate-600">
                {bay.replace("_", " ")}
              </span>
              {ev ? (
                <>
                  <span
                    className="inline-flex w-fit items-center gap-1 rounded-full px-1.5 text-[10px] font-bold text-white"
                    style={{ background: heat }}
                  >
                    {ev.risk_level}
                  </span>
                  <span className="text-[10px] leading-tight text-slate-500">
                    {ev.behavior_class.replace(/_/g, " ").slice(0, 22)}
                  </span>
                </>
              ) : (
                <span className="text-[10px] text-slate-300">no events</span>
              )}
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-[11px] text-slate-400">
        Color = worst detected risk per bay; larger dots on the replay page.
      </p>
    </section>
  );
}