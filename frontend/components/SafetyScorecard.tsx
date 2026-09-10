"use client";

// Team safety scorecard: headline KPIs + gamification elements derived from
// the persisted event log. Shows safety score, streak, badges, and team rank.

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

// Gamification: safety score out of 100, streak, and earned badges
function computeSafetyScore(events: RiskEvent[]): {
  score: number;
  streak: number;
  badges: { icon: string; label: string; color: string }[];
} {
  if (events.length === 0) {
    return { score: 100, streak: 0, badges: [{ icon: "🌟", label: "Clean start", color: "bg-emerald-100 text-emerald-700" }] };
  }

  const avgRisk = events.reduce((s, e) => s + e.risk_score, 0) / events.length;
  const criticalCount = events.filter((e) => e.risk_level === "critical").length;
  const highCount = events.filter((e) => e.risk_level === "high").length;

  // Score: starts at 100, loses points per risk
  let score = Math.max(0, Math.round(100 - avgRisk * 50 - criticalCount * 5 - highCount * 2));

  // Streak: consecutive low-risk events at the end of the list (newest first)
  const sorted = [...events].sort((a, b) => b.timestamp_sec - a.timestamp_sec);
  let streak = 0;
  for (const ev of sorted) {
    if (RISK_ORDER[ev.risk_level] <= RISK_ORDER.medium) streak++;
    else break;
  }

  // Badges
  const badges: { icon: string; label: string; color: string }[] = [];
  if (score >= 90) badges.push({ icon: "🏆", label: "Safety champion", color: "bg-yellow-100 text-yellow-700" });
  if (score >= 70) badges.push({ icon: "🛡️", label: "Guardian", color: "bg-blue-100 text-blue-700" });
  if (criticalCount === 0 && events.length > 0) badges.push({ icon: "✅", label: "Zero critical", color: "bg-emerald-100 text-emerald-700" });
  if (streak >= 5) badges.push({ icon: "🔥", label: `${streak} safe streak`, color: "bg-orange-100 text-orange-700" });
  if (events.length >= 10) badges.push({ icon: "📊", label: "Data-driven", color: "bg-purple-100 text-purple-700" });
  if (badges.length === 0) badges.push({ icon: "📈", label: "Improving", color: "bg-slate-100 text-slate-600" });

  return { score, streak, badges };
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
  const { score, streak, badges } = computeSafetyScore(events);

  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-6">
      {kpi("Events detected", String(events.length), `${today}`, "text-slate-900")}
      {kpi("High / critical", String(serious), "need follow-up", serious > 0 ? "text-orange-600" : "text-emerald-600")}
      {kpi("Avg risk score", events.length ? avg.toFixed(2) : "—", "0 (safe) -> 1 (critical)", "text-slate-900")}
      {kpi("Top behavior", topClass.replace(/_/g, " "), "most frequent", "text-indigo-600")}
      {/* Gamification KPIs */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Safety score</p>
        <p className={`mt-1 text-2xl font-bold ${score >= 80 ? "text-emerald-600" : score >= 50 ? "text-amber-600" : "text-red-600"}`}>
          {score}
          <span className="text-sm text-slate-400">/100</span>
        </p>
        <div className="mt-1 h-1.5 w-full rounded-full bg-slate-100">
          <div
            className={`h-full rounded-full transition-all ${score >= 80 ? "bg-emerald-500" : score >= 50 ? "bg-amber-500" : "bg-red-500"}`}
            style={{ width: `${score}%` }}
          />
        </div>
      </div>
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Badges</p>
        <div className="mt-1 flex flex-wrap gap-1">
          {badges.map((b) => (
            <span key={b.label} className={`inline-flex items-center gap-0.5 rounded-full px-2 py-0.5 text-[10px] font-semibold ${b.color}`}>
              {b.icon} {b.label}
            </span>
          ))}
        </div>
        {streak > 0 && (
          <p className="mt-1 text-[11px] text-slate-400">
            🔥 {streak} consecutive safe events
          </p>
        )}
      </div>
    </div>
  );
}