"use client";

// Event timeline: chronological time-axis view of events, showing the flow
// of safety incidents over time with risk-colored markers and expandable details.

import { useEffect, useState } from "react";
import { getEvents, RiskEvent, RISK_STYLES, RISK_ORDER } from "@/lib/api";
import Link from "next/link";

export default function EventTimeline() {
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    getEvents(100).then(setEvents).catch(() => {});
  }, []);

  // Group events by approximate time bucket (every 5 seconds of clip time)
  const buckets = new Map<number, RiskEvent[]>();
  for (const ev of events) {
    const bucket = Math.floor(ev.timestamp_sec / 5) * 5;
    if (!buckets.has(bucket)) buckets.set(bucket, []);
    buckets.get(bucket)!.push(ev);
  }
  const sortedBuckets = [...buckets.entries()].sort((a, b) => a[0] - b[0]);

  // Find max risk for scaling
  const maxRisk = Math.max(...events.map((e) => e.risk_score), 0.01);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Event timeline</h2>
        <span className="text-xs text-slate-400">{events.length} events</span>
      </header>

      {sortedBuckets.length === 0 ? (
        <p className="rounded-lg bg-slate-50 px-3 py-8 text-center text-sm text-slate-400">
          No events to display on the timeline.
        </p>
      ) : (
        <div className="relative">
          {/* Vertical time axis line */}
          <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-200" />

          <div className="space-y-3">
            {sortedBuckets.map(([t, evs]) => {
              const worst = evs.reduce((w, e) =>
                RISK_ORDER[e.risk_level] > RISK_ORDER[w.risk_level] ? e : w
              );
              const s = RISK_STYLES[worst.risk_level];
              return (
                <div key={t} className="relative flex gap-3 pl-1">
                  {/* Time marker dot */}
                  <div
                    className="relative z-10 mt-1 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full ring-2 ring-white"
                    style={{ background: s.heat }}
                  >
                    <span className="text-[10px] font-bold text-white">
                      {evs.length}
                    </span>
                  </div>

                  {/* Time label */}
                  <div className="flex-shrink-0 pt-0.5 text-xs font-medium text-slate-500">
                    {t.toFixed(0)}s
                  </div>

                  {/* Events in this bucket */}
                  <div className="flex-1 space-y-1">
                    {evs.map((ev) => {
                      const es = RISK_STYLES[ev.risk_level];
                      const isOpen = expanded === ev.event_id;
                      return (
                        <div key={ev.event_id} className="group">
                          <button
                            onClick={() =>
                              setExpanded(isOpen ? null : ev.event_id)
                            }
                            className="w-full rounded-lg border border-slate-100 bg-slate-50/60 px-3 py-1.5 text-left transition-colors hover:bg-slate-100"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-2">
                                <span
                                  className="h-1.5 w-8 rounded-full"
                                  style={{
                                    background: `linear-gradient(90deg, ${es.heat}, transparent)`,
                                    opacity: 0.3 + (ev.risk_score / maxRisk) * 0.7,
                                  }}
                                />
                                <span className="text-xs font-medium text-slate-700">
                                  {ev.behavior_class.replace(/_/g, " ")}
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] text-slate-400">
                                  {ev.zone_id}
                                </span>
                                <span
                                  className={`rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${es.badge}`}
                                >
                                  {ev.risk_level}
                                </span>
                              </div>
                            </div>
                          </button>

                          {/* Expanded detail */}
                          {isOpen && (
                            <div className="mt-1 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600 shadow-sm">
                              <p className="mb-2">{ev.justification}</p>
                              <div className="flex items-center gap-3 text-[11px] text-slate-400">
                                <span>Score: {ev.risk_score.toFixed(2)}</span>
                                <span>
                                  Confidence:{" "}
                                  {ev.confidence != null
                                    ? `${(ev.confidence * 100).toFixed(0)}%`
                                    : "N/A"}
                                </span>
                                <span>
                                  Physics:{" "}
                                  {ev.physics_valid ? "verified" : "unverified"}
                                </span>
                              </div>
                              <Link
                                href={`/replay/${ev.event_id}`}
                                className="mt-2 inline-block text-indigo-600 hover:underline"
                              >
                                View replay →
                              </Link>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
