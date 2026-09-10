"use client";

// Team safety scorecard: headline KPIs derived from the persisted event log.

import { useEffect, useState } from "react";
import { getEvents, RiskEvent, RISK_ORDER } from "@/lib/api";

function kpi(label: string, value: string, sub: string, accent: string) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
      <p className="mt-1 text-xs text-slate-400">{sub}</p>
    </div>
  );
}

export default function SafetyScorecard() {
  const [events, setEvents] = useState<RiskEvent[]>([]);

  useEffect(() => {
    getEvents().then(setEvents).catch(() => {});
  }, []);

  const serious = events.filter((e) => RISK_ORDER[e.risk_level] >= RISK_ORDER.high).length;
  const avg =
    events.length > 0
      ? events.reduce((s, e) => s + e.risk_score, 0) / events.length
      : 0;

  const top = new Map<string, number>();
  for (const ev of events) top.set(ev.behavior_class, (top.get(ev.behavior_class) ?? 0) + 1);
  const topClass = [...top.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";

  const today = new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" });

  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      {kpi("Events detected", String(events.length), `${today}`, "text-slate-900")}
      {kpi("High / critical", String(serious), "need follow-up", serious > 0 ? "text-orange-600" : "text-emerald-600")}
      {kpi("Avg risk score", events.length ? avg.toFixed(2) : "—", "0 (safe) → 1 (critical)", "text-slate-900")}
      {kpi("Top behavior", topClass.replace(/_/g, " "), "most frequent", "text-indigo-600")}
    </div>
  );
}