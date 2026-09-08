"use client";

// Behavior trends: count of events per behavior class (Recharts bar chart).
// Loaded with { ssr: false } so Recharts never renders on the server (it can
// crash during static pre-render), which keeps `next build` clean.

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getEvents, RiskEvent } from "@/lib/api";

const Chart = dynamic(() => import("./BehaviorChartInner"), { ssr: false });

export default function BehaviorTrends() {
  const [events, setEvents] = useState<RiskEvent[]>([]);

  useEffect(() => {
    getEvents().then(setEvents).catch(() => {});
  }, []);

  const byBehavior = new Map<string, number>();
  for (const ev of events) {
    byBehavior.set(ev.behavior_class, (byBehavior.get(ev.behavior_class) ?? 0) + 1);
  }
  const data = [...byBehavior.entries()]
    .map(([name, count]) => ({ name: name.replace(/_/g, " "), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-slate-700">Behavior trends</h2>
      {data.length === 0 ? (
        <p className="rounded-lg bg-slate-50 px-3 py-8 text-center text-sm text-slate-400">
          Ingest a clip to populate behavior trends.
        </p>
      ) : (
        <Chart data={data} />
      )}
    </section>
  );
}