"use client";

// Improvement trend line chart: daily risk aggregate over time.
// Shows whether warehouse handling is improving or regressing.

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getTrend, TrendDay } from "@/lib/api";

function ChartInner({ data }: { data: TrendDay[] }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="day" tick={{ fontSize: 10 }} stroke="#94a3b8" />
        <YAxis tick={{ fontSize: 10 }} stroke="#94a3b8" />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 8 }}
          formatter={(value: number, name: string) =>
            name === "avg_risk" ? value.toFixed(3) : value
          }
        />
        <Line
          type="monotone"
          dataKey="n"
          name="events"
          stroke="#6366f1"
          strokeWidth={2}
          dot={{ r: 3 }}
        />
        <Line
          type="monotone"
          dataKey="serious"
          name="high/critical"
          stroke="#f97316"
          strokeWidth={2}
          dot={{ r: 3 }}
        />
        <Line
          type="monotone"
          dataKey="avg_risk"
          name="avg_risk"
          stroke="#10b981"
          strokeWidth={1.5}
          strokeDasharray="4 2"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

const Chart = dynamic(() => Promise.resolve(ChartInner), { ssr: false });

export default function TrendLineChart() {
  const [data, setData] = useState<TrendDay[]>([]);
  const [windowDays, setWindowDays] = useState(7);

  useEffect(() => {
    getTrend(windowDays)
      .then((r) => setData(r.days))
      .catch(() => {});
  }, [windowDays]);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Improvement trend</h2>
        <div className="flex items-center gap-2">
          {[7, 14, 30].map((d) => (
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

      {data.length === 0 ? (
        <p className="rounded-lg bg-slate-50 px-3 py-8 text-center text-sm text-slate-400">
          Ingest clips over multiple days to see the improvement trend.
        </p>
      ) : (
        <>
          <Chart data={data} />
          <div className="mt-2 flex items-center gap-4 text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-indigo-500" /> total events
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-orange-500" /> high/critical
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-0.5 w-3 border-t-2 border-dashed border-emerald-500" /> avg risk
            </span>
          </div>
        </>
      )}
    </section>
  );
}
